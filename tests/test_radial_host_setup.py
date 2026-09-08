"""Exact host setup under batched NumPy geometry and optional nogil crop packing."""
from dataclasses import replace
import math
import unittest
from unittest import mock

import numpy as np

from XTA import cylindrical_projection as projection, cylindrical_cuda_projection as cuda, geometry


def sampled_shear_reference(view):
    result = np.zeros((view.num_slices, view.src_w), np.float64)
    columns = np.arange(view.src_w, dtype=np.float64)
    tangent = math.tan(math.radians(view.tilt_angle_deg))
    for shell, radius in enumerate(view.radial_radii):
        theta = np.remainder((float(view.radial_arc_origin) + columns) / radius, 2.0 * math.pi)
        offset = radius * (np.sin(theta) if view.tilt_direction == 'vertical' else np.cos(theta))
        result[shell] = tangent * offset
    return result


class RadialHostSetupTests(unittest.TestCase):
    def test_batched_shear_matches_original_bits_for_tails_tilts_and_chunk_sizes(self):
        for base in ('transverse', 'sagittal', 'coronal'):
            initial = geometry.get_view_infos(37, 41, 43, cartesian_views=(), radial_views=(base,),
                radial_min_radius=.01, radial_patch_size=33)[0]
            for width in (1, 3, 7, 16, 31, 32, 33, 65, 3072):
                for direction in ('vertical', 'horizontal'):
                    for angle in (-45., -30., .01, 23., 30., 45.):
                        view = replace(initial, src_w=width, radial_tilted_source=True,
                                       tilt_direction=direction, tilt_angle_deg=angle)
                        plan = projection._build_radial_plane_plan(view,
                            np.asarray(geometry.radial_global_radii(view)), (37, 41, 43))
                        expected = sampled_shear_reference(view)
                        for budget in (1, 7 * width, 256 * 1024):
                            with mock.patch.object(projection, '_RADIAL_METADATA_CHUNK_VALUES', budget):
                                actual = projection._radial_projection_metadata(view,
                                    (view.num_slices, 29, 17), (37, 41, 43), plan)[2]
                            # Include signed zeros, not only numerical equality.
                            np.testing.assert_array_equal(actual.view(np.uint64), expected.view(np.uint64))

    def test_cropped_interval_packing_matches_numpy_with_empty_shells_and_split_rows(self):
        rng = np.random.default_rng(515)
        source = rng.integers(0, 256, (17, 29, 37), dtype=np.uint8)
        boxes = np.zeros((17, 4), np.int64)
        boxes[1] = (0, 29, 0, 37)
        boxes[3] = (3, 17, 7, 9)
        boxes[7] = (1, 2, 3, 36)
        boxes[11] = (2, 23, 13, 37)
        sizes = (boxes[:, 1] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 2])
        offsets = np.zeros(18, np.uint64)
        np.cumsum(sizes, dtype=np.uint64, out=offsets[1:])
        expected = np.concatenate([source[shell, y0:y1, x0:x1].reshape(-1)
            for shell, (y0, y1, x0, x1) in enumerate(boxes)])
        packers = [cuda._pack_radial_source_block]
        if cuda._pack_radial_source_block_compiled is not None:
            packers.append(cuda._pack_radial_source_block_compiled)
        for pack in packers:
            for chunk in (1, 7, 37, 131, expected.size + 1):
                blocks = []
                for first in range(0, expected.size, chunk):
                    # Sentinels ensure no writes escape the admitted output slice.
                    data = np.full(min(chunk, expected.size - first) + 2, 199, np.uint8)
                    pack(source, boxes, offsets, first, data[1:-1])
                    self.assertEqual((int(data[0]), int(data[-1])), (199, 199))
                    blocks.append(data[1:-1])
                np.testing.assert_array_equal(np.concatenate(blocks), expected)
            pack(source, boxes, offsets, expected.size, np.empty(0, np.uint8))


if __name__ == '__main__':
    unittest.main()
