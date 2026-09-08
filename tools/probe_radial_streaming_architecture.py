"""Small exactness probe for a shell-chunk owner pipeline; not a speed benchmark.

Consume complete, already-cleaned 2D radius masks directly into a source-space
bitset. This deliberately slow reference groups the EXISTING pull relation by
owning shell. It neither retains a native 3D mask nor uses legacy D1 splat math.
The production pipeline does not import or enable this probe.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from XTA import cylindrical_projection as projection, geometry


class ShellChunkPullReference:
    """Proof of exact, disjoint shell ownership with explicit completion coverage."""
    def __init__(self, view, mask_shape, output_shape, *, expected_shells=None):
        self.view = view
        self.mask_shape = tuple(mask_shape)
        self.output_shape = tuple(output_shape)
        if math.prod(self.output_shape) > 100_000 or math.prod(self.mask_shape) > 100_000:
            raise ValueError('This is a bounded architecture proof, not a production projector')
        radii = np.asarray(geometry.radial_global_radii(view))
        self.plan = projection._build_radial_plane_plan(view, radii, self.output_shape)
        metadata = projection._radial_projection_metadata(view,
            (view.num_slices, *self.mask_shape), self.output_shape, self.plan)
        (self.centers, self.ideal, self.sampled, self.row_map, self.column_map,
         self.stack_length, self.vertical) = metadata
        # A production CSR bucket table can group these positions in linear time.
        # Retain the original periodic-column CSR and exact host shear tables.
        self.pixels_by_shell = tuple(np.flatnonzero(self.plan.shell_index == shell)
                                     for shell in range(view.num_slices))
        self.words = np.zeros((math.prod(self.output_shape) + 31) // 32, np.uint32)
        self.expected = np.zeros(view.num_slices, bool)
        shells = list(range(view.num_slices) if expected_shells is None else expected_shells)
        if len(set(shells)) != len(shells) or any(s < 0 or s >= view.num_slices for s in shells):
            raise ValueError('Invalid expected shell coverage')
        self.expected[shells] = True
        self.coverage = np.zeros(view.num_slices, bool)
        self.closed = False
        self.owned_output_positions_visited = 0

    def consume(self, shell, mask):
        if self.closed:
            raise RuntimeError('Contribution is sealed')
        shell = int(shell)
        if not 0 <= shell < self.view.num_slices or not self.expected[shell]:
            raise ValueError('Shell belongs to a different owner')
        if self.coverage[shell]:
            raise ValueError('Duplicate shell coverage')
        source = np.asarray(mask)
        if source.ndim != 2 or source.shape != self.mask_shape:
            raise ValueError('A complete 2D processing-grid shell mask is required')
        out_t, out_h, out_w = self.output_shape
        plane_w = self.plan.plane_shape[1]
        for pixel in self.pixels_by_shell[shell]:
            py, px = divmod(int(pixel), plane_w)
            ideal_height = self.centers - self.ideal[py if self.vertical else px]
            valid = (ideal_height >= 0.) & (ideal_height <= self.stack_length - 1)
            values = np.zeros(len(self.centers), bool)
            first, stop = self.plan.column_offsets[pixel:pixel + 2]
            for column in self.plan.native_columns[first:stop]:
                height = self.centers - self.sampled[shell, column]
                native_rows = np.rint(np.clip(height, 0., self.stack_length - 1)).astype(np.int64)
                native_rows -= int(self.view.radial_height_origin)
                inside = valid & (native_rows >= 0) & (native_rows < self.view.src_h)
                values[inside] |= source[self.row_map[native_rows[inside]], self.column_map[column]] != 0
            stack = np.flatnonzero(values)
            if self.plan.base_id == 0:
                flat = stack * (out_h * out_w) + pixel
            elif self.plan.base_id == 1:
                flat = (py * out_h + stack) * out_w + px
            else:
                flat = (py * out_h + px) * out_w + stack
            np.bitwise_or.at(self.words, flat // 32,
                            np.left_shift(np.uint32(1), (flat % 32).astype(np.uint32)))
            self.owned_output_positions_visited += len(self.centers)
        # Empty masks still satisfy coverage. No reference to the input survives.
        self.coverage[shell] = True

    def seal(self):
        if not np.array_equal(self.coverage, self.expected):
            raise RuntimeError('Cannot publish incomplete shell coverage')
        self.closed = True
        self.words.flags.writeable = False
        return self.words


def decode_words(words, shape):
    flat = np.arange(math.prod(shape), dtype=np.int64)
    return ((words[flat // 32] >> (flat % 32).astype(np.uint32)) & 1).astype(np.uint8).reshape(shape)


def oracle(source, view, shape):
    radii = np.asarray(geometry.radial_global_radii(view))
    return np.stack([projection._pull_radial_chunk(source, view, radii, shape, z, 0,
        shape[1] * shape[2]).reshape(shape[1:]) for z in range(shape[0])])


def run_matrix():
    rng = np.random.default_rng(142543)
    report = {'cases': [], 'limits': 'CPU semantic proof on small grids, not a performance estimate, '
        'CUDA implementation, cleanup qualification or production pipeline integration.'}
    total = 0
    for base in ('transverse', 'sagittal', 'coronal'):
        for minimum in (.01, .7):
            views = geometry.get_view_infos(7, 9, 11, cartesian_views=(), radial_views=(base,),
                radial_min_radius=minimum, radial_patch_size=6)
            for initial in views:
                for direction, angle in (('', 0.), ('vertical', -30.), ('vertical', 30.),
                                         ('horizontal', -30.), ('horizontal', 30.)):
                    view = replace(initial, radial_tilted_source=bool(direction),
                                   tilt_direction=direction, tilt_angle_deg=angle)
                    for mask_shape in ((6, 6), (4, 5)):
                        source = rng.integers(0, 2, (view.num_slices, *mask_shape), dtype=np.uint8) * np.uint8(255)
                        for shape in ((7, 9, 11), (5, 7, 8), (9, 11, 13)):
                            expected = oracle(source, view, shape)
                            owner = ShellChunkPullReference(view, mask_shape, shape)
                            for shell in rng.permutation(view.num_slices):
                                # Passing a copy models an input lease that can retire
                                # immediately after consumption. Overwrite it to prove
                                # the owner retained no borrowed mask data.
                                borrowed = source[shell].copy()
                                owner.consume(shell, borrowed)
                                borrowed[:] = 0
                            actual = decode_words(owner.seal(), shape)
                            np.testing.assert_array_equal(actual, expected)
                            owners = [ShellChunkPullReference(view, mask_shape, shape,
                                expected_shells=range(rank, view.num_slices, 2)) for rank in range(2)]
                            for shell in rng.permutation(view.num_slices):
                                owners[shell % 2].consume(shell, source[shell])
                            reduced = owners[0].seal() | owners[1].seal()
                            np.testing.assert_array_equal(decode_words(reduced, shape), expected)
                            total += expected.size
                            report['cases'].append({'view': view.name, 'base': base, 'min_radius': minimum,
                                'direction': direction, 'angle': angle, 'mask_shape': mask_shape, 'output_shape': shape,
                                'packed_bytes': owner.words.nbytes,
                                'source_native_3d_bytes_not_retained': source.nbytes,
                                'sha256': hashlib.sha256(actual.tobytes()).hexdigest()})
    report.update(all_exact=True, case_count=len(report['cases']), output_voxels_compared=total,
                  shuffled_shell_order=True, two_owner_reduction=True, retired_inputs_overwritten=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = run_matrix()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({k: v for k, v in report.items() if k != 'cases'}), flush=True)


if __name__ == '__main__':
    main()
