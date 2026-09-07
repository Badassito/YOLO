"""Independent source-coordinate checks for radius-swept intrinsic patches."""
from __future__ import annotations

import contextlib
from dataclasses import replace
import io
import math
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest import mock

import numpy as np

from XTA import assembly, backprojection, geometry, tta_terminal, cylindrical_projection as cp
from XTA.config import TiltedViewGroup, resolve_tilted_view_groups
from XTA.cylindrical_projection import backproject_radial_volume_to_volume
from XTA.interpolation import RawBBoxMaskStore, write_raw_bbox_mask_store
from XTA.runtime import close_memmap_array


def shells(base='transverse', *, shape=(7, 9, 11), size=5, minimum=0.7, tilt=False):
    groups = [TiltedViewGroup((base,), (23.0,), ('vertical', 'horizontal'))] if tilt else []
    return [v for v in geometry.get_view_infos(
        *shape, cartesian_views=(), radial_views=(('tilted_' if tilt else '') + base,),
        radial_min_radius=minimum, radial_patch_size=size, tilt_groups=groups,
    ) if v.family == 'radial']


def scalar_oracle(data, view, shape):
    """Deliberately slow scalar nearest-shell/periodic-occurrence reference."""
    out = np.zeros(shape, np.uint8)
    count = int(math.ceil(view.radial_max_radius - view.radial_min_radius)) + 1
    radii = [view.radial_min_radius + i * (view.radial_max_radius - view.radial_min_radius) / (count - 1)
             if count > 1 else view.radial_min_radius for i in range(count)]
    work = (view.full_t, view.full_h, view.full_w)
    for position in np.ndindex(shape):
        t, y, x = [(i + .5) * a / b - .5 for i, a, b in zip(position, work, shape)]
        if view.radial_base_view == 'transverse':
            stack, py, px, length = t, y, x, work[0]
        elif view.radial_base_view == 'sagittal':
            stack, py, px, length = y, t, x, work[1]
        else:
            stack, py, px, length = x, t, y, work[2]
        dx, dy = px - view.center_x, py - view.center_y
        distance = math.hypot(dx, dy)
        if not view.radial_min_radius <= distance <= view.radial_max_radius:
            continue
        if view.radial_tilted_source:
            stack -= math.tan(math.radians(view.tilt_angle_deg)) * (dy if view.tilt_direction == 'vertical' else dx)
        true_height = stack
        if not 0 <= true_height <= length - 1:
            continue
        chosen = min(range(count), key=lambda i: abs(radii[i] - distance))
        frame = chosen - view.radial_shell_start
        if not 0 <= frame < data.shape[0]:
            continue
        radius = radii[chosen]
        period = 2 * math.pi * radius
        arc = (math.atan2(dy, dx) % (2 * math.pi)) * radius
        first = math.floor((view.radial_arc_origin - arc - 1) / period)
        last = math.ceil((view.radial_arc_origin + view.src_w - arc + 1) / period)
        for k in range(first, last + 1):
            col = round(arc + k * period - view.radial_arc_origin)
            if not 0 <= col < view.src_w:
                continue
            # Compose the inverse at this actual sampled shell point rather
            # than the ideal source voxel's pre-quantization planar position.
            source_stack = {'transverse': t, 'sagittal': y, 'coronal': x}[view.radial_base_view]
            if view.radial_tilted_source:
                sample_theta = ((view.radial_arc_origin + col) / radius) % (2 * math.pi)
                sample_axis = radius * (math.sin(sample_theta) if view.tilt_direction == 'vertical' else math.cos(sample_theta))
                source_stack -= math.tan(math.radians(view.tilt_angle_deg)) * sample_axis
            row = round(max(0, min(length - 1, source_stack))) - view.radial_height_origin
            if 0 <= row < view.src_h and data[frame, row, col]:
                out[position] = 1
                break
    return out


class CylindricalProjectionTests(unittest.TestCase):
    def setUp(self):
        cpu_only = mock.patch.dict('os.environ', {'YOLO_TTA_GPU_RADIAL_BACKPROJECT': '0'})
        cpu_only.start()
        self.addCleanup(cpu_only.stop)
        stdout = contextlib.redirect_stdout(io.StringIO())
        stdout.__enter__()
        self.addCleanup(stdout.__exit__, None, None, None)
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.serial = 0

    def project(self, data, view, shape=None):
        self.serial += 1
        with contextlib.redirect_stdout(io.StringIO()):
            result = backproject_radial_volume_to_volume(
                data, view, self.root / f'{self.serial}.dat', 'shell test',
                out_shape_tyx=shape, reserve_bytes=0,
            )
        try:
            return np.asarray(result).copy()
        finally:
            close_memmap_array(result)

    def test_dense_all_ones_cover_annulus_for_all_axes(self):
        shape = (7, 9, 11)
        for base in ('transverse', 'sagittal', 'coronal'):
            views = shells(base, shape=shape)
            union = np.zeros(shape, np.uint8)
            for view in views:
                union |= self.project(np.ones((view.num_slices, view.src_h, view.src_w), np.uint8), view)
            zz, yy, xx = np.indices(shape)
            py, px = {'transverse': (yy, xx), 'sagittal': (zz, xx), 'coronal': (zz, yy)}[base]
            distance = np.hypot(px - views[0].center_x, py - views[0].center_y)
            expected = ((distance >= .7) & (distance <= views[0].radial_max_radius)).astype(np.uint8)
            np.testing.assert_array_equal(union, expected, err_msg=base)

    def test_random_masks_match_scalar_oracle_with_subset_offsets_and_tilt(self):
        rng = np.random.default_rng(940)
        for base in ('transverse', 'sagittal', 'coronal'):
            for tilted in (False, True):
                views = shells(base, shape=(5, 7, 9), size=4, tilt=tilted)
                for view in views[::max(1, len(views) // 4)]:
                    data = (rng.random((view.num_slices, view.src_h, view.src_w)) < .23).astype(np.uint8)
                    for target in ((5, 7, 9), (3, 5, 7)):
                        with self.subTest(view=view.name, shape=target):
                            np.testing.assert_array_equal(self.project(data, view, target), scalar_oracle(data, view, target))
        self.assertTrue(any(v.radial_shell_start > 0 for v in views))

    def test_periodic_duplicates_are_orred_and_inner_shell_does_not_leak(self):
        view = shells(shape=(3, 7, 7), size=12, minimum=1.0)[0]
        data = np.zeros((view.num_slices, 12, 12), np.uint8)
        # At radius one, theta zero occurs at columns 0 and round(2*pi)=6.
        data[0, 1, 6] = 1
        actual = self.project(data, view)
        self.assertEqual(actual[1, 3, 4], 1)
        self.assertEqual(actual[1, 3, 3], 0)  # explicitly excluded central core
        data[0, 1, 0] = 1
        np.testing.assert_array_equal(self.project(data, view), scalar_oracle(data, view, actual.shape))

    def test_tiny_circumference_single_shell_and_bounded_chunks(self):
        view = shells(shape=(3, 5, 5), size=8, minimum=.1)[0]
        data = np.zeros((view.num_slices, 8, 8), np.uint8)
        data[:, :3, 7] = 1
        with mock.patch('XTA.cylindrical_projection._PULL_CHUNK_VOXELS', 3):
            np.testing.assert_array_equal(self.project(data, view), scalar_oracle(data, view, (3, 5, 5)))
        single = shells(shape=(3, 5, 5), size=5, minimum=2)[0]
        self.assertEqual(single.num_slices, 1)
        ones = np.ones((1, 5, 5), np.uint8)
        np.testing.assert_array_equal(self.project(ones, single), scalar_oracle(ones, single, (3, 5, 5)))

    def test_source_space_sink_and_tile_parent_coordinates(self):
        view = shells(shape=(3, 7, 9), size=5, minimum=1)[0]
        data = np.zeros((view.num_slices, 5, 5), np.uint8)
        # A cleaned nested tile has already been mapped into its intrinsic parent patch.
        data[:, 1:3, 3:5] = 1
        blocks = []
        result = assembly.project_view_volume_to_orthogonal_volume(
            data, view, self.root / 'unused.dat', 'tile parent shell',
            out_shape_tyx=(3, 7, 9), sink_only=True,
            projection_block_callback=lambda z, b: blocks.append((z, b.copy())),
        )
        self.assertIsInstance(result, backprojection.SinkOnlyProjectionResult)
        self.assertEqual([z + i for z, block in blocks for i in range(len(block))], [0, 1, 2])
        np.testing.assert_array_equal(np.concatenate([b for _, b in blocks]), scalar_oracle(data, view, result.shape))
        self.assertFalse((self.root / 'unused.dat').exists())
        self.assertFalse(assembly.view_interpolation_wrap_axis(view))

    def test_terminal_tta_collapse_projects_radial_once(self):
        view = shells(shape=(3, 7, 9), size=5, minimum=1)[0]
        a = np.zeros((view.num_slices, 5, 5), np.uint8)
        b = a.copy()
        a[:, 1:3, :2] = 1
        b[:, 1:3, 3:] = 1
        va, vb = [replace(view, name=view.name + suffix, physical_view_name=view.name, tta_aug_id=suffix)
                  for suffix in ('a', 'b')]
        with contextlib.redirect_stdout(io.StringIO()):
            _, _, result = tta_terminal.finalize_physical_view_volume_group(
                model_name='model', physical_view=view, variant_volumes=((va, a), (vb, b)),
                out_path=self.root / 'terminal.dat', out_shape_tyx=(3, 7, 9), workers=1,
                collapse_variants=lambda *_args, **_kwargs: {'model': {view.name: a | b}},
            )
        try:
            np.testing.assert_array_equal(result, scalar_oracle(a | b, view, (3, 7, 9)))
        finally:
            close_memmap_array(result)

    def test_nrrd_refs_and_empty_bridge_have_source_geometry(self):
        view = shells(shape=(3, 7, 9), size=5, minimum=1)[0]
        data = np.ones((view.num_slices, 5, 5), np.uint8)
        sink = mock.Mock()
        with mock.patch.object(assembly, 'final_source_output_shape', return_value=(3, 7, 9)), \
                mock.patch.object(assembly, 'nrrd_layer_sink', return_value=sink), \
                contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            for source in ('fullframe', 'tile'):
                ref = assembly.materialize_nrrd_view_layer(
                    data, model_name='model', view=view, source=source, mask_kind='yolo',
                    tile_config_id='nested' if source == 'tile' else '',
                    temp_dir=self.root, force_path_backed_store=True,
                )
                self.assertEqual(ref.shape, (3, 7, 9))
                self.assertEqual(ref.view_family, 'radial')
                store = RawBBoxMaskStore.open(ref.path)
                try:
                    decoded = np.stack([store.decode_slice(i) for i in range(3)])
                finally:
                    store.close()
                np.testing.assert_array_equal(decoded, scalar_oracle(data, view, ref.shape))
            empty = self.root / 'empty.cvol'
            write_raw_bbox_mask_store(np.zeros_like(data), empty, desc='empty', workers=1)
            ref = assembly.materialize_interpolation_component_nrrd_view_layer(
                empty, added_voxels=0, model_name='model', view=view, source='fullframe',
                pass_index=1, interpolation_walk_back_index=1, interpolation_candidate_index=1,
                stage='interpolation', description='empty shell bridge', temp_dir=self.root,
                workers=1, keep_temp=True,
            )
            self.assertEqual(ref.shape, (3, 7, 9))
            store = RawBBoxMaskStore.open(ref.path)
            try:
                self.assertFalse(any(np.any(store.decode_slice(i)) for i in range(3)))
            finally:
                store.close()
        self.assertEqual(sink.submit_layer.call_count, 3)

    def test_bad_depth_and_azimuthal_inputs_are_rejected_without_output(self):
        view = shells()[0]
        with self.assertRaisesRegex(ValueError, 'depth'):
            self.project(np.zeros((view.num_slices + 1, 5, 5), np.uint8), view)
        with self.assertRaisesRegex(ValueError, 'Radial view'):
            self.project(np.zeros((view.num_slices, 5, 5), np.uint8), replace(view, family='azimuthal'))

    def test_actual_render_foreground_covers_tilted_annulus_boundary_matrix(self):
        # This exercises real forward-sampling support, rather than merely
        # assigning one to padded model pixels. The +/-45 coronal (8,9,10), S7
        # case used to lose two source-face voxels because inverse height was
        # selected before quantizing the radius/periodic angular sample.
        cases = 0
        for shape, size in (((1, 11, 13), 4), ((5, 8, 9), 3), ((8, 9, 10), 7),
                            ((7, 7, 7), 4), ((2, 4, 6), 2)):
            volume = np.full(shape, 255, np.uint8)
            z, y, x = np.indices(shape)
            for base in ('transverse', 'sagittal', 'coronal'):
                plane_dims = {'transverse': shape[1:], 'sagittal': (shape[0], shape[2]),
                              'coronal': shape[:2]}[base]
                if min(plane_dims) <= 1:
                    continue
                for minimum in (.01, size / (4 * math.pi)):
                    if minimum > (min(plane_dims) - 1) / 2:
                        continue
                    for direction in ('vertical', 'horizontal'):
                        for angle in (1, 30, 45):
                            built = geometry.get_view_infos(
                                *shape, cartesian_views=(), radial_views=('tilted_' + base,),
                                radial_min_radius=minimum, radial_patch_size=size,
                                tilt_groups=resolve_tilted_view_groups([f'{base}:{angle}:{direction}']),
                            )
                            for sign in (-1, 1):
                                views = [v for v in built if v.family == 'radial' and np.sign(v.tilt_angle_deg) == sign]
                                union = np.zeros(shape, np.uint8)
                                for view in views:
                                    mask = np.stack([geometry.get_view_frame_by_index(volume, view, i) > 0
                                                     for i in range(view.num_slices)])
                                    def consume(first, block):
                                        union[first:first + len(block)] |= block
                                    backproject_radial_volume_to_volume(
                                        mask, view, self.root / 'unused.dat', 'coverage matrix',
                                        sink_only=True, projection_block_callback=consume,
                                    )
                                view = views[0]
                                stack, py, px, length = {
                                    'transverse': (z, y, x, shape[0]),
                                    'sagittal': (y, z, x, shape[1]),
                                    'coronal': (x, z, y, shape[2]),
                                }[base]
                                dx, dy = px - view.center_x, py - view.center_y
                                radius = np.hypot(dx, dy)
                                height = stack - np.tan(np.radians(view.tilt_angle_deg)) * (dy if direction == 'vertical' else dx)
                                wanted = ((radius >= minimum) & (radius <= view.radial_max_radius)
                                          & (height >= 0) & (height <= length - 1))
                                with self.subTest(shape=shape, size=size, base=base, minimum=minimum,
                                                  direction=direction, angle=angle, sign=sign):
                                    np.testing.assert_array_equal(union.astype(bool), wanted)
                                cases += 1
        self.assertEqual(cases, 312)

    def test_factored_kernel_matches_unchanged_pull_for_random_reduced_and_restored_grids(self):
        rng = np.random.default_rng(202)
        for base in ('transverse', 'sagittal', 'coronal'):
            views = shells(base, shape=(7, 9, 11), size=6, minimum=.1, tilt=True)
            for view in views[::max(1, len(views) // 5)]:
                for processing_size in (4, 6):
                    data = (rng.random((view.num_slices, processing_size, processing_size)) < .17).astype(np.uint8)
                    for output_shape in ((7, 9, 11), (5, 7, 8), (9, 11, 13)):
                        radii = np.asarray(geometry.radial_global_radii(view))
                        expected = np.stack([cp._pull_radial_chunk(
                            data, view, radii, output_shape, z, 0, output_shape[1] * output_shape[2],
                        ).reshape(output_shape[1:]) for z in range(output_shape[0])])
                        with self.subTest(view=view.name, processing_size=processing_size, output_shape=output_shape):
                            with mock.patch.object(cp, '_OUTPUT_BLOCK_BYTES', output_shape[1] * output_shape[2]):
                                blocks = []
                                cp.backproject_radial_volume_to_volume(
                                    data, view, self.root / 'unused.dat', 'factored oracle',
                                    out_shape_tyx=output_shape, workers=3, sink_only=True,
                                    projection_block_callback=lambda z, block: blocks.append((z, block.copy())),
                                )
                            np.testing.assert_array_equal(np.concatenate([b for _, b in blocks]), expected)

    def test_plane_cache_is_readonly_and_reuses_tilt_and_height_metadata(self):
        cp.clear_radial_plane_plan_cache()
        view = shells(shape=(7, 9, 11), size=6)[0]
        radii = np.asarray(geometry.radial_global_radii(view))
        first, hit = cp._radial_plane_plan(view, radii, (7, 9, 11))
        self.assertFalse(hit)
        changed = replace(view, radial_tilted_source=True, tilt_angle_deg=-23,
                          tilt_direction='horizontal', radial_height_origin=1)
        second, hit = cp._radial_plane_plan(changed, radii, (7, 9, 11))
        self.assertTrue(hit)
        self.assertIs(first, second)
        self.assertFalse(first.shell_index.flags.writeable)
        self.assertFalse(first.native_columns.flags.writeable)
        shifted, hit = cp._radial_plane_plan(replace(view, center_x=view.center_x + .1), radii, (7, 9, 11))
        self.assertFalse(hit)
        self.assertIsNot(shifted, first)

    def test_source_bboxes_skip_empty_frames_without_changing_results(self):
        view = shells(shape=(7, 9, 11), size=6)[0]
        data = np.zeros((view.num_slices, 6, 6), np.uint8)
        data[::2, 1:4, 2:5] = 1
        boxes = np.zeros((view.num_slices, 4), np.int64)
        boxes[::2] = (1, 4, 2, 5)
        expected = self.project(data, view)
        blocks = []
        cp.backproject_radial_volume_to_volume(
            data, view, self.root / 'unused.dat', 'bbox oracle', workers=3,
            known_slice_bboxes=boxes, sink_only=True,
            projection_block_callback=lambda z, block: blocks.append(block.copy()),
        )
        np.testing.assert_array_equal(np.concatenate(blocks), expected)
        boxes[0, 1] = 7
        with self.assertRaisesRegex(ValueError, 'bounding boxes'):
            cp.backproject_radial_volume_to_volume(data, view, self.root / 'invalid.dat', 'bad bbox', known_slice_bboxes=boxes)

    def test_callback_failure_waits_for_running_blocks_before_returning_borrowed_input(self):
        view = shells(shape=(7, 9, 11), size=6)[0]
        data = np.ones((view.num_slices, 6, 6), np.uint8)
        live = [0, 0]
        lock = threading.Lock()
        started_together = threading.Barrier(3)
        def project(*args):
            count = args[-3]
            if not count:
                return np.empty((0, 9, 11), np.uint8)
            with lock:
                live[0] += 1
                live[1] = max(live[1], live[0])
            try:
                started_together.wait(timeout=3.0)
                time.sleep(.02)
                return np.ones((count, 9, 11), np.uint8)
            finally:
                with lock:
                    live[0] -= 1
        with mock.patch.object(cp, '_project_radial_block', side_effect=project), \
                mock.patch.object(cp, '_numba', object()), \
                mock.patch.object(cp, '_OUTPUT_BLOCK_BYTES', 99), \
                mock.patch.object(cp, '_cpu_count', return_value=3):
            with self.assertRaisesRegex(RuntimeError, 'sink rejected'):
                cp.backproject_radial_volume_to_volume(
                    data, view, self.root / 'unused.dat', 'failure', workers=3, sink_only=True,
                    projection_block_callback=mock.Mock(side_effect=RuntimeError('sink rejected')),
                )
        self.assertEqual(live[0], 0)
        self.assertGreater(live[1], 1)
        self.assertLessEqual(live[1], 3)
        self.assertTrue(data.all())


if __name__ == '__main__':
    unittest.main()
