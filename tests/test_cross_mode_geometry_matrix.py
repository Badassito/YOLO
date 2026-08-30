from __future__ import annotations

import math
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from tools.smoke_import import install_stubs


install_stubs()

from XTA import geometry
from XTA import pta
from XTA.pta_config import parse_pta_args
from XTA.unification.contracts import (
    DataRole,
    FrameAddress,
    PipelineMode,
    RenderItem,
)
from XTA.unification.sampling import (
    build_forward_raster_plan,
    forward_sampling_policy,
)


def _rotation_matrix(
    center: tuple[float, float], angle_deg: float, scale: float
) -> np.ndarray:
    """Small NumPy stand-in for the unavailable development OpenCV build."""

    alpha = math.cos(math.radians(float(angle_deg))) * float(scale)
    beta = math.sin(math.radians(float(angle_deg))) * float(scale)
    center_x, center_y = (float(value) for value in center)
    return np.asarray(
        [
            [
                alpha,
                beta,
                (1.0 - alpha) * center_x - beta * center_y,
            ],
            [
                -beta,
                alpha,
                beta * center_x + (1.0 - alpha) * center_y,
            ],
        ],
        dtype=np.float64,
    )


def _warp_affine(
    source: np.ndarray,
    matrix: np.ndarray,
    dsize: tuple[int, int],
    *,
    flags: int,
    borderMode: int,
    borderValue: int,
) -> np.ndarray:
    """Deterministic nearest/linear CPU affine used by both compared paths.

    This intentionally implements only the OpenCV contract used by the forward
    renderers: a source-to-destination matrix, constant-zero boundary, uint8
    two-dimensional input, nearest or bilinear interpolation.
    """

    del borderMode
    src = np.asarray(source)
    if src.ndim != 2:
        raise AssertionError(f"matrix test expects one grayscale plane, got {src.shape}")
    out_w, out_h = (int(value) for value in dsize)
    supplied = np.eye(3, dtype=np.float64)
    supplied[:2, :3] = np.asarray(matrix, dtype=np.float64).reshape(2, 3)
    # OpenCV's WARP_INVERSE_MAP bit declares that M already maps destination
    # pixels to the source. Without it, M is the forward source-to-destination
    # transform and the sampler inverts it internally.
    inverse = supplied if (int(flags) & 16) else np.linalg.inv(supplied)
    yy, xx = np.indices((out_h, out_w), dtype=np.float64)
    src_x = inverse[0, 0] * xx + inverse[0, 1] * yy + inverse[0, 2]
    src_y = inverse[1, 0] * xx + inverse[1, 1] * yy + inverse[1, 2]

    if (int(flags) & 7) == 0:  # INTER_NEAREST
        x_idx = np.rint(src_x).astype(np.int64)
        y_idx = np.rint(src_y).astype(np.int64)
        valid = (
            (x_idx >= 0)
            & (x_idx < int(src.shape[1]))
            & (y_idx >= 0)
            & (y_idx < int(src.shape[0]))
        )
        out = np.full((out_h, out_w), int(borderValue), dtype=src.dtype)
        out[valid] = src[y_idx[valid], x_idx[valid]]
        return np.ascontiguousarray(out)

    x0 = np.floor(src_x).astype(np.int64)
    y0 = np.floor(src_y).astype(np.int64)
    dx = src_x - x0
    dy = src_y - y0
    accum = np.zeros((out_h, out_w), dtype=np.float64)
    weight_sum = np.zeros((out_h, out_w), dtype=np.float64)
    for y_offset, wy in ((0, 1.0 - dy), (1, dy)):
        for x_offset, wx in ((0, 1.0 - dx), (1, dx)):
            x_idx = x0 + int(x_offset)
            y_idx = y0 + int(y_offset)
            weight = wx * wy
            valid = (
                (x_idx >= 0)
                & (x_idx < int(src.shape[1]))
                & (y_idx >= 0)
                & (y_idx < int(src.shape[0]))
            )
            accum[valid] += (
                src[y_idx[valid], x_idx[valid]].astype(np.float64) * weight[valid]
            )
            weight_sum[valid] += weight[valid]
    if int(borderValue) != 0:
        accum += (1.0 - weight_sum) * float(borderValue)
    return np.ascontiguousarray(
        np.clip(np.rint(accum), 0.0, 255.0).astype(src.dtype)
    )


class CrossModeGeometryMatrixTests(unittest.TestCase):
    """Exact same-backend PTA/TTA forward-render qualification matrix."""

    VOLUME_SHAPE = (5, 6, 7)
    VIEW_CASES = (
        (
            "cartesian",
            ("--enable_cartesian", "transverse"),
        ),
        (
            "radial",
            ("--enable_radial", "transverse:60"),
        ),
        (
            "tilted_cartesian",
            ("--enable_tilted", "sagittal:20:vertical"),
        ),
        (
            "tilted_radial",
            (
                "--enable_tilted",
                "coronal:20:horizontal",
                "--enable_radial",
                "tilted_coronal:60",
            ),
        ),
    )

    def setUp(self) -> None:
        t_idx, y_idx, x_idx = np.indices(self.VOLUME_SHAPE)
        self.intensity = np.asarray(
            (47 * t_idx + 19 * y_idx + 7 * x_idx + 3) % 251,
            dtype=np.uint8,
        )
        self.categorical = np.asarray(
            (((5 * t_idx + 3 * y_idx + x_idx) % 7) < 2) * 9,
            dtype=np.uint8,
        )
        self._cv_patches = (
            mock.patch.object(
                geometry.cv2,
                "getRotationMatrix2D",
                side_effect=_rotation_matrix,
            ),
            mock.patch.object(
                geometry.cv2,
                "warpAffine",
                side_effect=_warp_affine,
            ),
            mock.patch.object(geometry.cv2, "INTER_NEAREST", 0, create=True),
            mock.patch.object(geometry.cv2, "INTER_LINEAR", 1, create=True),
            mock.patch.object(geometry.cv2, "WARP_INVERSE_MAP", 16, create=True),
            mock.patch.object(geometry.cv2, "BORDER_CONSTANT", 0, create=True),
        )
        for patcher in self._cv_patches:
            patcher.start()

    def tearDown(self) -> None:
        for patcher in reversed(self._cv_patches):
            patcher.stop()

    def _compile_view(self, case_name: str, arguments: tuple[str, ...]) -> pta.ViewInfo:
        config = parse_pta_args(["--input", "dataset", *arguments])
        adapted, compiled = pta.compile_v18_pta_views(
            t_dim=self.VOLUME_SHAPE[0],
            h=self.VOLUME_SHAPE[1],
            w=self.VOLUME_SHAPE[2],
            config=config,
            radial_native_raster=0,
        )
        self.assertEqual(
            [view.shared_view for view in adapted],
            list(compiled.views),
            msg=f"{case_name}: PTA adapters must retain the exact TTA view objects",
        )
        if case_name == "cartesian":
            matches = [
                view
                for view in adapted
                if view.shared_view is not None
                and not geometry.is_radial_view(view.shared_view)
                and not geometry.is_tilted_view(view.shared_view)
            ]
        elif case_name == "radial":
            matches = [
                view
                for view in adapted
                if view.shared_view is not None
                and geometry.is_radial_view(view.shared_view)
                and not geometry.is_tilted_radial_view(view.shared_view)
            ]
        elif case_name == "tilted_cartesian":
            matches = [
                view
                for view in adapted
                if view.shared_view is not None
                and geometry.is_tilted_view(view.shared_view)
                and float(view.shared_view.tilt_angle_deg) > 0.0
            ]
        else:
            matches = [
                view
                for view in adapted
                if view.shared_view is not None
                and geometry.is_tilted_radial_view(view.shared_view)
                and float(view.shared_view.tilt_angle_deg) > 0.0
            ]
        self.assertTrue(matches, msg=f"{case_name}: requested physical view was not compiled")
        return matches[0]

    @staticmethod
    def _frame_index(view: pta.ViewInfo) -> int:
        return min(1, max(0, int(view.num_slices) - 1))

    @staticmethod
    def _variant(format_token: str, direction: str = "forward") -> pta.ChannelVariant:
        variants = pta.expand_channel_variants(
            pta.resolve_channel_formats([format_token])
        )
        return next(value for value in variants if value.order_name == direction)

    def _build_plan(
        self,
        *,
        view: pta.ViewInfo,
        imgsz: int,
        variant: pta.ChannelVariant,
        out_dir: Path,
        with_tiles: bool,
    ) -> pta.RenderPlan:
        affine = pta.build_affine(
            view.src_w,
            view.src_h,
            0.0,
            view.pad_mode,
            int(imgsz),
            shared_view=view.shared_view,
        )
        return pta.build_render_plan(
            view=view,
            aff=affine,
            tag=f"{view.name}_{variant.tag_token}_{imgsz}",
            out_dir=out_dir,
            stem="sample",
            tile_configs=(pta.TileConfig(4, 3, "s4_st3"),) if with_tiles else (),
            save_overlay=False,
            imgsz=int(imgsz),
            label_enabled=True,
            channel_variant=variant,
            publish_images=False,
            publish_labels=False,
        )

    def _assert_plan_binding(self, plan: pta.RenderPlan) -> None:
        self.assertIsNotNone(plan.canonical_plan)
        assert plan.canonical_plan is not None
        current_policy = forward_sampling_policy()
        self.assertEqual(plan.canonical_plan.mode, PipelineMode.PTA)
        self.assertEqual(plan.canonical_plan.sampling_policy.digest, current_policy.digest)
        self.assertEqual(plan.stats["sampling_policy_digest"], current_policy.digest)
        self.assertEqual(plan.stats["canonical_plan_digest"], plan.canonical_plan.digest)
        self.assertEqual(
            plan.canonical_plan.physical_view_id,
            geometry.physical_view_name(plan.view.shared_view),
        )

        expected = build_forward_raster_plan(
            mode="pta",
            physical_view_id=geometry.physical_view_name(plan.view.shared_view),
            angle_deg=float(plan.aff.angle_deg),
            channel_token=str(plan.channel_variant.format_token),
            channel_kind=str(plan.channel_variant.kind),
            channel_count=int(plan.channel_variant.channel_count),
            channel_stride=int(plan.channel_variant.stride),
            channel_offsets=tuple(int(value) for value in plan.channel_variant.offsets),
            channel_direction=str(plan.channel_variant.order_name),
            output_shape=(int(plan.aff.out_h), int(plan.aff.out_w)),
            metadata={
                "runtime_view_id": str(plan.view.name),
                "runtime_job_id": str(plan.tag),
                "runtime_kind": "fullframe",
            },
        )
        self.assertEqual(plan.canonical_plan.digest, expected.digest)

        # Mode is intentionally part of the plan identity; the sampling policy
        # is intentionally not. Cross-mode execution therefore shares the
        # policy digest without pretending the publication plans are identical.
        tta_plan = build_forward_raster_plan(
            mode="tta",
            physical_view_id=expected.physical_view_id,
            angle_deg=expected.in_plane_variant.angle_deg,
            channel_token=expected.channel_variant.layout.token,
            channel_kind=expected.channel_variant.layout.kind,
            channel_count=expected.channel_variant.layout.channel_count,
            channel_stride=expected.channel_variant.layout.stride,
            channel_offsets=expected.channel_variant.offsets,
            channel_direction=expected.channel_variant.direction,
            output_shape=expected.output_shape,
            metadata=dict(expected.metadata),
        )
        self.assertEqual(tta_plan.sampling_policy.digest, expected.sampling_policy.digest)
        self.assertNotEqual(tta_plan.digest, expected.digest)

        self.assertEqual(
            plan.stats["canonical_tile_plan_digests"],
            [
                tile.canonical_plan.digest
                for tile in plan.tile_layout
                if tile.canonical_plan is not None
            ],
        )
        for tile in plan.tile_layout:
            self.assertIsNotNone(tile.shared_job)
            self.assertIsNotNone(tile.canonical_plan)
            assert tile.canonical_plan is not None
            self.assertEqual(tile.canonical_plan.sampling_policy.digest, current_policy.digest)
            self.assertEqual(
                tile.canonical_plan.output_shape,
                (int(tile.out_h), int(tile.out_w)),
            )
            self.assertEqual(tile.canonical_plan.tile_layout.tile_size, tile.cfg.tile_size)
            self.assertEqual(tile.canonical_plan.tile_layout.tile_stride, tile.cfg.tile_stride)

    @staticmethod
    def _direct_full_plane(
        volume: np.ndarray,
        plan: pta.RenderPlan,
        source_idx: int,
        *,
        categorical: bool,
        mirror_u: bool = False,
    ) -> np.ndarray:
        assert plan.view.shared_view is not None
        renderer = (
            geometry.render_categorical_frame_on_grid
            if categorical
            else geometry.render_intensity_frame_on_grid
        )
        return renderer(
            volume,
            plan.view.shared_view,
            int(source_idx),
            M_src_to_out=plan.aff.M_src_to_out,
            M_out_to_src=plan.aff.M_out_to_src,
            output_height=int(plan.aff.out_h),
            output_width=int(plan.aff.out_w),
            mirror_radial_u=bool(mirror_u),
        )

    @staticmethod
    def _direct_tile_plane(
        volume: np.ndarray,
        plan: pta.RenderPlan,
        tile: pta.RenderTileItem,
        source_idx: int,
        *,
        categorical: bool,
        mirror_u: bool = False,
    ) -> np.ndarray:
        assert plan.view.shared_view is not None
        assert tile.shared_job is not None
        renderer = (
            geometry.render_categorical_dense_tile_for_job
            if categorical
            else geometry.render_dense_tile_frame_for_job
        )
        return renderer(
            volume,
            plan.view.shared_view,
            tile.shared_job,
            int(source_idx),
            mirror_radial_u=bool(mirror_u),
        )

    def test_intensity_and_categorical_fullframe_and_tile_matrix(self) -> None:
        """4 families x 2 output grids x 2 roles x full/tile are exact."""

        gray = self._variant("gray")
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            for case_name, arguments in self.VIEW_CASES:
                view = self._compile_view(case_name, arguments)
                frame_idx = self._frame_index(view)
                for imgsz in (0, 5):
                    with self.subTest(case=case_name, imgsz=imgsz):
                        plan = self._build_plan(
                            view=view,
                            imgsz=imgsz,
                            variant=gray,
                            out_dir=root / case_name / str(imgsz),
                            with_tiles=True,
                        )
                        self._assert_plan_binding(plan)
                        actual = pta.render_plan_frame_source(
                            volume=self.intensity,
                            mask=self.categorical,
                            plan=plan,
                            idx=frame_idx,
                            need_canvas=True,
                        )

                        expected_image = self._direct_full_plane(
                            self.intensity,
                            plan,
                            frame_idx,
                            categorical=False,
                        )
                        expected_mask = self._direct_full_plane(
                            self.categorical,
                            plan,
                            frame_idx,
                            categorical=True,
                        )
                        np.testing.assert_array_equal(actual.img_full, expected_image)
                        np.testing.assert_array_equal(actual.mask_full, expected_mask)
                        self.assertEqual(actual.img_full.dtype, np.uint8)
                        self.assertEqual(actual.mask_full.dtype, np.uint8)
                        self.assertLessEqual(set(np.unique(actual.mask_full).tolist()), {0, 1})
                        expected_full_shape = (
                            (view.src_h, view.src_w)
                            if imgsz == 0
                            else (imgsz, imgsz)
                        )
                        self.assertEqual(actual.img_full.shape, expected_full_shape)

                        self.assertEqual(
                            set(actual.tile_arrays),
                            {tile.tile_tag for tile in plan.tile_layout},
                        )
                        for tile in plan.tile_layout:
                            actual_image, actual_mask = actual.tile_arrays[tile.tile_tag]
                            expected_image = self._direct_tile_plane(
                                self.intensity,
                                plan,
                                tile,
                                frame_idx,
                                categorical=False,
                            )
                            expected_mask = self._direct_tile_plane(
                                self.categorical,
                                plan,
                                tile,
                                frame_idx,
                                categorical=True,
                            )
                            np.testing.assert_array_equal(actual_image, expected_image)
                            np.testing.assert_array_equal(actual_mask, expected_mask)
                            self.assertLessEqual(
                                set(np.unique(actual_mask).tolist()), {0, 1}
                            )

    def _expected_channel_full(
        self,
        plan: pta.RenderPlan,
        center_idx: int,
    ) -> np.ndarray:
        variant = plan.channel_variant
        assert plan.view.shared_view is not None
        center_address = geometry.channel_view_slice_source(
            plan.view.shared_view, int(center_idx)
        )
        if variant.kind in {"gray", "rgb"}:
            plane = self._direct_full_plane(
                self.intensity,
                plan,
                center_address[0],
                categorical=False,
                mirror_u=center_address[1],
            )
            if variant.kind == "gray":
                return plane
            return np.ascontiguousarray(np.repeat(plane[:, :, None], 3, axis=2))
        addresses = tuple(
            geometry.channel_view_slice_source(
                plan.view.shared_view, int(center_idx) + int(offset)
            )
            for offset in variant.offsets
        )
        return np.ascontiguousarray(
            np.stack(
                [
                    self._direct_full_plane(
                        self.intensity,
                        plan,
                        source_idx,
                        categorical=False,
                        mirror_u=mirror_u,
                    )
                    for source_idx, mirror_u in addresses
                ],
                axis=2,
            )
        )

    def _expected_channel_tile(
        self,
        plan: pta.RenderPlan,
        tile: pta.RenderTileItem,
        center_idx: int,
    ) -> np.ndarray:
        variant = plan.channel_variant
        assert plan.view.shared_view is not None
        center_address = geometry.channel_view_slice_source(
            plan.view.shared_view, int(center_idx)
        )
        if variant.kind in {"gray", "rgb"}:
            plane = self._direct_tile_plane(
                self.intensity,
                plan,
                tile,
                center_address[0],
                categorical=False,
                mirror_u=center_address[1],
            )
            if variant.kind == "gray":
                return plane
            return np.ascontiguousarray(np.repeat(plane[:, :, None], 3, axis=2))
        addresses = tuple(
            geometry.channel_view_slice_source(
                plan.view.shared_view, int(center_idx) + int(offset)
            )
            for offset in variant.offsets
        )
        return np.ascontiguousarray(
            np.stack(
                [
                    self._direct_tile_plane(
                        self.intensity,
                        plan,
                        tile,
                        source_idx,
                        categorical=False,
                        mirror_u=mirror_u,
                    )
                    for source_idx, mirror_u in addresses
                ],
                axis=2,
            )
        )

    def test_gray_rgb_and_custom_forward_reverse_channel_matrix(self) -> None:
        """Every family/grid/channel direction uses direct TTA plane renders."""

        variants = (
            self._variant("gray"),
            self._variant("RGB"),
            self._variant("C3S1", "forward"),
            self._variant("C3S1", "reverse"),
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            for case_name, arguments in self.VIEW_CASES:
                view = self._compile_view(case_name, arguments)
                # Center zero forces the first custom plane across a Radial seam.
                center_idx = 0 if case_name in {"radial", "tilted_radial"} else 1
                for imgsz in (0, 5):
                    for variant in variants:
                        with self.subTest(
                            case=case_name,
                            imgsz=imgsz,
                            channel=variant.format_token,
                            direction=variant.order_name,
                        ):
                            plan = self._build_plan(
                                view=view,
                                imgsz=imgsz,
                                variant=variant,
                                out_dir=(
                                    root
                                    / case_name
                                    / str(imgsz)
                                    / variant.tag_token
                                ),
                                with_tiles=True,
                            )
                            self._assert_plan_binding(plan)
                            full, canvas = pta.render_channel_formatted_images(
                                volume=self.intensity,
                                plan=plan,
                                idx=center_idx,
                                need_canvas=False,
                            )
                            self.assertIsNone(canvas)
                            np.testing.assert_array_equal(
                                full,
                                self._expected_channel_full(plan, center_idx),
                            )
                            self.assertTrue(plan.tile_layout)
                            tile = plan.tile_layout[0]
                            actual_tile = pta.render_shared_tile_images(
                                volume=self.intensity,
                                plan=plan,
                                tile=tile,
                                idx=center_idx,
                            )
                            np.testing.assert_array_equal(
                                actual_tile,
                                self._expected_channel_tile(plan, tile, center_idx),
                            )

    def test_radial_seam_wrap_mirror_and_multiwrap_parity(self) -> None:
        view = self._compile_view(
            "radial", ("--enable_radial", "transverse:60")
        )
        assert view.shared_view is not None
        shared_view = view.shared_view
        count = int(shared_view.num_slices)
        self.assertEqual(count, 3)
        self.assertEqual(geometry.channel_view_slice_source(shared_view, -1), (2, True))
        self.assertEqual(geometry.channel_view_slice_source(shared_view, 3), (0, True))
        self.assertEqual(geometry.channel_view_slice_source(shared_view, -4), (2, False))
        self.assertEqual(geometry.channel_view_slice_source(shared_view, 6), (0, False))

        with tempfile.TemporaryDirectory() as temp_dir:
            plan = self._build_plan(
                view=view,
                imgsz=0,
                variant=self._variant("C3S1", "forward"),
                out_dir=Path(temp_dir),
                with_tiles=True,
            )
            nominal_image = self._direct_full_plane(
                self.intensity, plan, count - 1, categorical=False
            )
            mirrored_image = self._direct_full_plane(
                self.intensity,
                plan,
                count - 1,
                categorical=False,
                mirror_u=True,
            )
            nominal_mask = self._direct_full_plane(
                self.categorical, plan, count - 1, categorical=True
            )
            mirrored_mask = self._direct_full_plane(
                self.categorical,
                plan,
                count - 1,
                categorical=True,
                mirror_u=True,
            )
            np.testing.assert_array_equal(mirrored_image, nominal_image[:, ::-1])
            np.testing.assert_array_equal(mirrored_mask, nominal_mask[:, ::-1])
            np.testing.assert_array_equal(
                pta._shared_render_full_intensity(
                    self.intensity,
                    view,
                    count - 1,
                    plan.aff,
                    mirror_radial_u=True,
                ),
                mirrored_image,
            )
            np.testing.assert_array_equal(
                pta._shared_render_full_mask(
                    self.categorical,
                    view,
                    count - 1,
                    plan.aff,
                    mirror_radial_u=True,
                ),
                mirrored_mask,
            )

            full, _canvas = pta.render_channel_formatted_images(
                volume=self.intensity,
                plan=plan,
                idx=0,
                need_canvas=False,
            )
            np.testing.assert_array_equal(full[:, :, 0], mirrored_image)
            tile = plan.tile_layout[0]
            tile_stack = pta.render_shared_tile_images(
                volume=self.intensity,
                plan=plan,
                tile=tile,
                idx=0,
            )
            expected_mirrored_tile = self._direct_tile_plane(
                self.intensity,
                plan,
                tile,
                count - 1,
                categorical=False,
                mirror_u=True,
            )
            np.testing.assert_array_equal(tile_stack[:, :, 0], expected_mirrored_tile)

    def test_pta_dataset_sink_receives_the_exact_canonical_frame_and_item(self) -> None:
        view = self._compile_view(
            "cartesian", ("--enable_cartesian", "transverse")
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            plan = self._build_plan(
                view=view,
                imgsz=5,
                variant=self._variant("gray"),
                out_dir=root,
                with_tiles=False,
            )
            assert plan.canonical_plan is not None
            image = np.ascontiguousarray(
                self._direct_full_plane(
                    self.intensity,
                    plan,
                    1,
                    categorical=False,
                ),
                dtype=np.uint8,
            )
            candidate = pta.OutputCandidate(
                order=0,
                volume_name="sample",
                parent_view_tag=plan.tag,
                output_tag=plan.tag,
                item_key="full",
                frame_idx=1,
                is_tile=False,
                label_enabled=True,
                physical_view_id=plan.canonical_plan.physical_view_id,
                presentation_variant_id="channel:gray:forward",
                geometry_item_id="full",
                channel_format="gray",
                channel_kind="gray",
                channel_offsets=(0,),
            )
            with mock.patch.object(pta, "write_image") as writer:
                batch = pta.publish_pta_candidate_image_batch(
                    cand=candidate,
                    image=image,
                    img_path=root / "sample.png",
                    canonical_plan=plan.canonical_plan,
                    png_compression=1,
                    jpeg_quality=95,
                )

        self.assertIs(batch.frames[0], image)
        self.assertIs(batch.items[0].frame, batch.frames[0])
        self.assertIs(batch.model_payload()[1], batch.frames)
        self.assertIs(writer.call_args.args[1], batch.frames[0])
        self.assertIs(batch.raster_plan, plan.canonical_plan)
        self.assertIsNotNone(batch.items[0].request)
        request = batch.items[0].request
        assert request is not None
        expected_request = RenderItem(
            plan=plan.canonical_plan,
            data_role=DataRole.INTENSITY,
            frame_address=FrameAddress(1),
            metadata={
                "physical_view_id": candidate.physical_view_id,
                "presentation_variant_id": candidate.presentation_variant_id,
                "geometry_item_id": candidate.geometry_item_id,
                "augmentation_index": 0,
                "augmentation_tag": None,
            },
        )
        self.assertEqual(request.item_id, expected_request.item_id)
        self.assertEqual(request.plan.digest, plan.canonical_plan.digest)


if __name__ == "__main__":
    unittest.main()
