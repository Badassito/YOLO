from __future__ import annotations

import importlib.util
import os
import sys
import tempfile
import threading
import types
import unittest
from unittest import mock

import numpy as np

from tools.smoke_import import install_stubs


def _available(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ModuleNotFoundError, ValueError):
        return False


if not all(_available(name) for name in ("cv2", "scipy", "tifffile", "tqdm")):
    install_stubs()

from XTA import cuda_finalization, topology


def _pairs(*values: tuple[int, int]) -> np.ndarray:
    return np.asarray(
        [(np.uint64(a) << np.uint64(32)) | np.uint64(b) for a, b in values],
        dtype=np.uint64,
    )


def _reference_keep_largest_26(
    source: np.ndarray, keep_n: int,
) -> tuple[np.ndarray, int, int, tuple[int, ...]]:
    """Independent scan/union reference for exact 26-connected keep-largest."""

    foreground = np.asarray(source) != 0
    labels = np.zeros(foreground.shape, dtype=np.int32)
    # Only neighbors that precede the current voxel in Z/Y/X scan order.  This
    # differs deliberately from the flood-fill fake used to emulate ndimage.label.
    prior_offsets = tuple(
        (dz, dy, dx)
        for dz in (-1, 0)
        for dy in (-1, 0, 1)
        for dx in (-1, 0, 1)
        if dz < 0 or dy < 0 or (dy == 0 and dx < 0)
    )
    parents = [0]

    def _new_label() -> int:
        parents.append(len(parents))
        return len(parents) - 1

    def _find(value: int) -> int:
        root = int(value)
        while int(parents[root]) != root:
            root = int(parents[root])
        while int(value) != root:
            parent = int(parents[int(value)])
            parents[int(value)] = root
            value = parent
        return root

    def _union(left: int, right: int) -> None:
        left_root, right_root = _find(left), _find(right)
        if left_root != right_root:
            parents[max(left_root, right_root)] = min(left_root, right_root)

    z_dim, height, width = (int(value) for value in foreground.shape)
    for z in range(z_dim):
        for y in range(height):
            for x in range(width):
                if not bool(foreground[z, y, x]):
                    continue
                neighbors: list[int] = []
                for dz, dy, dx in prior_offsets:
                    nz, ny, nx = z + dz, y + dy, x + dx
                    if 0 <= nz < z_dim and 0 <= ny < height and 0 <= nx < width:
                        label = int(labels[nz, ny, nx])
                        if label > 0:
                            neighbors.append(label)
                if not neighbors:
                    labels[z, y, x] = np.int32(_new_label())
                    continue
                label = min(neighbors)
                labels[z, y, x] = np.int32(label)
                for neighbor in neighbors:
                    _union(label, neighbor)

    areas: dict[int, int] = {}
    for z, y, x in np.argwhere(labels > 0):
        root = _find(int(labels[int(z), int(y), int(x)]))
        labels[int(z), int(y), int(x)] = np.int32(root)
        areas[root] = int(areas.get(root, 0) + 1)
    ranked = sorted(areas, key=lambda root: (-int(areas[root]), int(root)))
    keep_roots = set(ranked[: max(0, int(keep_n))])
    output = np.zeros(source.shape, dtype=np.uint8)
    if keep_roots:
        output[np.isin(labels, np.asarray(sorted(keep_roots), dtype=np.int32))] = np.uint8(1)
    return (
        output,
        int(len(areas)),
        int(sum(areas[root] for root in keep_roots)),
        tuple(sorted(int(value) for value in areas.values())),
    )


class _FakeDeviceArray(np.ndarray):
    """Small NumPy-backed stand-in preserving a logical CUDA device id."""

    def __new__(cls, values: object, device_index: int):
        result = np.asarray(values).view(cls)
        result.device_index = int(device_index)
        return result

    def __array_finalize__(self, source: object) -> None:
        self.device_index = int(getattr(source, "device_index", 0))


class _FakeGpuRuntime:
    def __init__(self, *, device_count: int = 2, peer_access: bool = True,
                 fail_label_device: int | None = None) -> None:
        self.device_count = int(device_count)
        self.peer_access = bool(peer_access)
        self.fail_label_device = fail_label_device
        self.local = threading.local()
        self.direct_peer_copies: list[tuple[int, int]] = []
        self.host_reads: list[int] = []
        self.pool_frees: list[int] = []

    @property
    def current_device(self) -> int:
        return int(getattr(self.local, "device", 0))

    def array(self, values: object, dtype: object = None) -> _FakeDeviceArray:
        if isinstance(values, _FakeDeviceArray):
            if int(values.device_index) != int(self.current_device):
                self.direct_peer_copies.append(
                    (int(self.current_device), int(values.device_index))
                )
            elif dtype is None or np.dtype(dtype) == values.dtype:
                return values
        return _FakeDeviceArray(
            np.array(np.asarray(values), dtype=dtype, copy=True), self.current_device,
        )

    def modules(self) -> dict[str, object]:
        owner = self

        class _Device:
            def __init__(self, device_index: int) -> None:
                self.device_index = int(device_index)
                self.prior = 0

            def __enter__(self) -> "_Device":
                self.prior = owner.current_device
                owner.local.device = int(self.device_index)
                return self

            def __exit__(self, *_args: object) -> None:
                owner.local.device = int(self.prior)

        class _Stream:
            ptr = 0

            @staticmethod
            def synchronize() -> None:
                return None

        class _Pool:
            @staticmethod
            def free_all_blocks() -> None:
                owner.pool_frees.append(int(owner.current_device))

        class _RuntimeApi:
            @staticmethod
            def deviceCanAccessPeer(_device: int, _peer: int) -> int:
                return int(owner.peer_access)

        cupy = types.ModuleType("cupy")
        cupy.uint8 = np.uint8
        cupy.uint32 = np.uint32
        cupy.uint64 = np.uint64
        cupy.int32 = np.int32
        cupy.int64 = np.int64
        cupy.bool_ = np.bool_
        cupy.asarray = lambda values, dtype=None: owner.array(values, dtype)
        cupy.asnumpy = self._asnumpy
        cupy.empty = lambda shape, dtype: _FakeDeviceArray(
            np.empty(shape, dtype=dtype), owner.current_device,
        )
        cupy.zeros = lambda shape, dtype: _FakeDeviceArray(
            np.zeros(shape, dtype=dtype), owner.current_device,
        )
        cupy.where = lambda condition, x, y: owner.array(np.where(condition, x, y))
        cupy.bincount = lambda values, minlength=0: owner.array(
            np.bincount(np.asarray(values, dtype=np.int64), minlength=int(minlength))
        )
        cupy.concatenate = lambda values: owner.array(
            np.concatenate([np.asarray(value) for value in values])
        )
        cupy.unique = lambda values: owner.array(np.unique(np.asarray(values)))
        cupy.get_default_memory_pool = lambda: _Pool()
        cupy.cuda = types.SimpleNamespace(
            Device=_Device,
            get_current_stream=lambda: _Stream(),
            runtime=_RuntimeApi(),
        )

        torch = types.ModuleType("torch")
        torch.cuda = types.SimpleNamespace(
            is_available=lambda: True,
            device_count=lambda: int(owner.device_count),
            device=lambda _device: _NoopContext(),
            mem_get_info=lambda _device: (512 * 1024 ** 3, 512 * 1024 ** 3),
        )

        ndimage = types.ModuleType("cupyx.scipy.ndimage")
        ndimage.label = self._label
        scipy = types.ModuleType("cupyx.scipy")
        scipy.ndimage = ndimage
        cupyx = types.ModuleType("cupyx")
        cupyx.scipy = scipy
        return {
            "torch": torch,
            "cupy": cupy,
            "cupyx": cupyx,
            "cupyx.scipy": scipy,
            "cupyx.scipy.ndimage": ndimage,
        }

    def _asnumpy(self, values: object) -> np.ndarray:
        if isinstance(values, _FakeDeviceArray):
            self.host_reads.append(int(values.device_index))
        return np.array(np.asarray(values), copy=True)

    def _label(self, foreground: object, *, structure: object) -> tuple[_FakeDeviceArray, int]:
        if self.fail_label_device == self.current_device:
            raise RuntimeError("injected fake CCL failure")
        structure_host = np.asarray(structure, dtype=bool)
        if structure_host.shape != (3, 3, 3):
            raise AssertionError(f"unexpected CCL structure {structure_host.shape}")
        fg = np.asarray(foreground, dtype=bool)
        labels = np.zeros(fg.shape, dtype=np.int32)
        next_id = 0
        for z in range(int(fg.shape[0])):
            for y in range(int(fg.shape[1])):
                for x in range(int(fg.shape[2])):
                    if not bool(fg[z, y, x]) or int(labels[z, y, x]) != 0:
                        continue
                    next_id += 1
                    labels[z, y, x] = np.int32(next_id)
                    pending = [(int(z), int(y), int(x))]
                    while pending:
                        cz, cy, cx = pending.pop()
                        for dz in (-1, 0, 1):
                            for dy in (-1, 0, 1):
                                for dx in (-1, 0, 1):
                                    if not bool(structure_host[dz + 1, dy + 1, dx + 1]):
                                        continue
                                    if dz == 0 and dy == 0 and dx == 0:
                                        continue
                                    nz, ny, nx = int(cz + dz), int(cy + dy), int(cx + dx)
                                    if not (
                                        0 <= nz < fg.shape[0]
                                        and 0 <= ny < fg.shape[1]
                                        and 0 <= nx < fg.shape[2]
                                    ):
                                        continue
                                    if bool(fg[nz, ny, nx]) and int(labels[nz, ny, nx]) == 0:
                                        labels[nz, ny, nx] = np.int32(next_id)
                                        pending.append((nz, ny, nx))
        return _FakeDeviceArray(labels, self.current_device), int(next_id)


class _NoopContext:
    def __enter__(self) -> "_NoopContext":
        return self

    def __exit__(self, *_args: object) -> None:
        return None


class CudaFinalizationContractTests(unittest.TestCase):
    @staticmethod
    def _numpy_metadata_scan(
        volume: np.ndarray, *, workers: int, source: str,
    ) -> topology.BinaryVolumeSliceMetadata:
        del workers
        arr = np.asarray(volume)
        slice_any = np.any(arr != 0, axis=(1, 2))
        slice_bboxes = np.zeros((int(arr.shape[0]), 4), dtype=np.int64)
        for z in np.flatnonzero(slice_any):
            rows, cols = np.nonzero(arr[int(z)] != 0)
            slice_bboxes[int(z)] = np.asarray(
                (int(rows.min()), int(rows.max()) + 1,
                 int(cols.min()), int(cols.max()) + 1),
                dtype=np.int64,
            )
        return topology.register_binary_volume_slice_metadata(
            volume, slice_any, slice_bboxes, source=str(source), exact=True,
        )

    @staticmethod
    def _physical_fixture() -> tuple[np.ndarray, np.ndarray]:
        source = np.zeros((10, 7, 8), dtype=np.uint8)
        # One 26-connected area-10 object crosses ordinary CCL rows, 4-slice block
        # boundaries, and the two-GPU shard boundary through diagonal Z neighbors.
        main_coords = (
            (1, 1), (1, 1), (1, 1), (2, 2), (1, 1),
            (2, 2), (2, 2), (2, 2), (2, 2), (1, 1),
        )
        for z, (y, x) in enumerate(main_coords):
            source[int(z), int(y), int(x)] = np.uint8(1)
        source[0, 5, 3:7] = np.uint8(1)    # area 4
        source[8:10, 5, 6] = np.uint8(1)   # area 2
        expected = np.zeros_like(source)
        for z, (y, x) in enumerate(main_coords):
            expected[int(z), int(y), int(x)] = np.uint8(1)
        return source, expected

    def _run_physical_fake(
        self, runtime: _FakeGpuRuntime, source: np.ndarray, temp_dir: str,
        *, keep_n: int = 1, devices: tuple[int, ...] = (0, 1),
        block_slices: int = 4,
    ) -> cuda_finalization.GpuTailKeepResult | None:
        with (
            mock.patch.dict(
                os.environ,
                {
                    "YOLO_TTA_V1803_GPU_RESIDENT_TAIL": "1",
                    "YOLO_TTA_V1803_GPU_RESIDENT_TAIL_REQUIRED": "0",
                    "YOLO_TTA_V1803_GPU_TAIL_RESERVE_GIB": "1",
                    "YOLO_TTA_V1803_GPU_TAIL_BLOCK_SLICES": str(int(block_slices)),
                },
                clear=False,
            ),
            mock.patch.dict(sys.modules, runtime.modules(), clear=False),
            mock.patch.object(
                cuda_finalization, "_configured_tail_devices", return_value=devices,
            ),
            mock.patch.object(
                topology,
                "scan_binary_volume_slice_metadata",
                side_effect=self._numpy_metadata_scan,
            ) as metadata_scan,
        ):
            result = cuda_finalization.try_apply_keep_largest_objects_multi_gpu(
                source, int(keep_n), temp_dir,
            )
            self._last_metadata_scan_call_count = int(metadata_scan.call_count)
            return result

    @staticmethod
    def _blockwise_matrix_fixture() -> np.ndarray:
        source = np.zeros((19, 15, 17), dtype=np.uint8)

        # Area 19: a diagonal 26-connected spine through every possible Z shard
        # boundary and through all tested local CCL block boundaries.
        for z in range(19):
            source[z, 1 + (z % 2), 1 + (z % 2)] = np.uint8(1)

        # Areas 13 and 8: separate diagonal spines with distinct starts/lengths.
        path = (0, 1, 2, 1)
        for offset, z in enumerate(range(2, 15)):
            delta = path[offset % len(path)]
            source[z, 6 + delta, 6 + delta] = np.uint8(1)
        for offset, z in enumerate(range(6, 14)):
            delta = offset % 2
            source[z, 11 + delta, 1 + delta] = np.uint8(1)

        # Areas 5 and 2 stay within one slice. All lanes are separated by at
        # least one empty Chebyshev shell, so their areas remain unique.
        source[17, 3, 10:15] = np.uint8(1)
        source[0, 12, 13:15] = np.uint8(1)
        return source

    def test_contiguous_partitions_cover_without_overlap(self) -> None:
        for z_dim, device_count in ((1, 8), (7, 4), (16, 4), (17, 8), (65, 8)):
            parts = cuda_finalization.contiguous_z_partitions(z_dim, device_count)
            self.assertEqual(parts[0].z0, 0)
            self.assertEqual(parts[-1].z1, z_dim)
            self.assertTrue(all(left.z1 == right.z0 for left, right in zip(parts, parts[1:])))
            self.assertLessEqual(max(part.slices for part in parts) - min(part.slices for part in parts), 1)

    def test_row_padded_pack_roundtrip_for_non_word_widths(self) -> None:
        rng = np.random.default_rng(17)
        for width in (1, 7, 31, 32, 33, 63, 64, 65):
            source = (rng.integers(0, 3, size=(3, 4, width), dtype=np.uint8) > 0).astype(np.uint8)
            packed = cuda_finalization.pack_binary_rows(source)
            self.assertEqual(packed.shape, (3, 4, cuda_finalization.row_word_count(width)))
            restored = cuda_finalization.unpack_binary_rows(packed, width)
            np.testing.assert_array_equal(restored, source)
            if width % 32:
                invalid = np.uint32(~cuda_finalization.row_tail_mask(width))
                self.assertFalse(bool(np.any(packed[..., -1] & invalid)))

    def test_artifact_plan_uses_only_selected_devices(self) -> None:
        artifact = cuda_finalization.artifact_plan_from_host(
            (9, 5, 7), (3, 1, 7, 5), source="fixture",
        )
        artifact.validate()
        self.assertEqual(tuple(shard.device_index for shard in artifact.shards), (3, 1, 7, 5))
        self.assertEqual(tuple((shard.z0, shard.z1) for shard in artifact.shards), ((0, 3), (3, 5), (5, 7), (7, 9)))

    def test_global_keep_decision_merges_cross_shard_object(self) -> None:
        shards = (
            cuda_finalization.KeepGraphShard(
                component_areas=np.asarray((0, 3, 2), dtype=np.int64),
                internal_pair_codes=np.zeros((0,), dtype=np.uint64),
            ),
            cuda_finalization.KeepGraphShard(
                component_areas=np.asarray((0, 4, 1), dtype=np.int64),
                internal_pair_codes=np.zeros((0,), dtype=np.uint64),
            ),
        )
        decision = cuda_finalization.resolve_keep_graph(
            shards,
            (cuda_finalization.CrossShardPairCodes(0, 1, _pairs((2, 1))),),
            keep_objects=1,
        )
        self.assertEqual(decision.num_objects, 3)
        self.assertEqual(decision.kept_objects, 1)
        self.assertEqual(decision.removed_objects, 2)
        self.assertEqual(decision.removed_voxels, 4)
        np.testing.assert_array_equal(
            decision.keep_by_shard_local_id[0], np.asarray((False, False, True)),
        )
        np.testing.assert_array_equal(
            decision.keep_by_shard_local_id[1], np.asarray((False, True, False)),
        )

    def test_internal_pairs_merge_components_before_area_ranking(self) -> None:
        decision = cuda_finalization.resolve_keep_graph(
            (
                cuda_finalization.KeepGraphShard(
                    component_areas=np.asarray((0, 2, 5, 7), dtype=np.int64),
                    internal_pair_codes=_pairs((1, 2)),
                ),
            ),
            (),
            keep_objects=1,
        )
        self.assertEqual(decision.num_objects, 2)
        # The merged 1+2 object has area 7, tied with component 3. Preserve the
        # current root-id-driven CPU ordering without assuming which tie wins.
        self.assertEqual(int(np.count_nonzero(decision.keep_by_shard_local_id[0][1:])), 1)
        self.assertEqual(decision.removed_voxels, 7)

    def test_graph_rejects_background_and_out_of_range_pair_ids(self) -> None:
        shard = cuda_finalization.KeepGraphShard(
            component_areas=np.asarray((0, 2), dtype=np.int64),
            internal_pair_codes=_pairs((0, 1)),
        )
        with self.assertRaisesRegex(ValueError, "high ids"):
            cuda_finalization.resolve_keep_graph((shard,), (), keep_objects=1)
        shard = cuda_finalization.KeepGraphShard(
            component_areas=np.asarray((0, 2), dtype=np.int64),
            internal_pair_codes=_pairs((1, 2)),
        )
        with self.assertRaisesRegex(ValueError, "low ids"):
            cuda_finalization.resolve_keep_graph((shard,), (), keep_objects=1)

    def test_numpy_cuda_fake_exercises_peer_boundary_and_transactional_apply(self) -> None:
        source, expected = self._physical_fixture()
        authority = source.copy()
        runtime = _FakeGpuRuntime(peer_access=True)
        with (
            tempfile.TemporaryDirectory() as temp_dir,
            mock.patch.object(cuda_finalization.gc, "collect", return_value=0) as collect,
        ):
            result = self._run_physical_fake(runtime, source, temp_dir)
            self.assertIsNotNone(result)
            assert result is not None
            np.testing.assert_array_equal(source, authority)
            np.testing.assert_array_equal(np.asarray(result.volume), expected)
            self.assertEqual(int(result.stats["num_objects"]), 3)
            self.assertEqual(int(result.stats["removed_objects"]), 2)
            self.assertEqual(int(result.stats["removed_voxels"]), 6)
            self.assertEqual(int(result.stats["peer_host_bounces"]), 0)
            self.assertGreater(int(result.stats["peer_bytes"]), 0)
            self.assertEqual(int(result.stats["ccl_blocks"]), 4)
            self.assertEqual(int(result.stats["pair_boundaries"]), 2)
            for timing_key in (
                "import_seconds", "discovery_seconds", "runtime_init_seconds",
                "metadata_seconds", "setup_seconds", "shard_stage_seconds",
            ):
                self.assertIn(timing_key, result.stats)
                self.assertGreaterEqual(float(result.stats[timing_key]), 0.0)
            # One outer cleanup collection remains; successful shard workers no longer
            # serialize one full-process collection apiece.
            self.assertEqual(collect.call_count, 1)
            self.assertEqual(self._last_metadata_scan_call_count, 1)
            self.assertEqual(runtime.direct_peer_copies, [(1, 0)])
            self.assertEqual(sorted(runtime.pool_frees), [0, 1])
            cuda_finalization._close_array(result.volume)

    def test_pre_registered_metadata_bypasses_fallback_scan(self) -> None:
        source, expected = self._physical_fixture()
        self._numpy_metadata_scan(source, workers=1, source="pre-registered test")
        runtime = _FakeGpuRuntime(peer_access=True)
        with tempfile.TemporaryDirectory() as temp_dir:
            result = self._run_physical_fake(runtime, source, temp_dir)
            self.assertIsNotNone(result)
            assert result is not None
            self.assertEqual(self._last_metadata_scan_call_count, 0)
            np.testing.assert_array_equal(np.asarray(result.volume), expected)
            cuda_finalization._close_array(result.volume)

    def test_numpy_cuda_fake_host_bounce_preserves_cross_shard_connectivity(self) -> None:
        source, expected = self._physical_fixture()
        runtime = _FakeGpuRuntime(peer_access=False)
        with tempfile.TemporaryDirectory() as temp_dir:
            result = self._run_physical_fake(runtime, source, temp_dir)
            self.assertIsNotNone(result)
            assert result is not None
            np.testing.assert_array_equal(np.asarray(result.volume), expected)
            self.assertEqual(int(result.stats["peer_host_bounces"]), 1)
            self.assertEqual(int(result.stats["peer_bytes"]), 0)
            self.assertEqual(runtime.direct_peer_copies, [])
            cuda_finalization._close_array(result.volume)

    def test_blockwise_3d_ccl_matches_independent_reference_across_layouts(self) -> None:
        source = self._blockwise_matrix_fixture()
        layouts = (
            # (block slices, device count, keep-largest N)
            (4, 1, 3),
            (4, 2, 2),
            (5, 3, 3),
            (7, 4, 1),
            (32, 8, 4),
        )
        for block_slices, device_count, keep_n in layouts:
            with self.subTest(
                block_slices=block_slices,
                device_count=device_count,
                keep_n=keep_n,
            ):
                expected, num_objects, kept_voxels, component_areas = _reference_keep_largest_26(
                    source, keep_n,
                )
                # The fixture's five object sizes are 19, 13, 8, 5, and 2. Unique
                # sizes ensure this reaches the blockwise implementation instead of
                # exercising its intentional equal-cutoff-tie fallback.
                self.assertEqual(num_objects, 5)
                self.assertEqual(component_areas, (2, 5, 8, 13, 19))
                runtime = _FakeGpuRuntime(
                    device_count=device_count, peer_access=True,
                )
                devices = tuple(range(device_count))
                with tempfile.TemporaryDirectory() as temp_dir:
                    result = self._run_physical_fake(
                        runtime,
                        source,
                        temp_dir,
                        keep_n=keep_n,
                        devices=devices,
                        block_slices=block_slices,
                    )
                    self.assertIsNotNone(result)
                    assert result is not None
                    np.testing.assert_array_equal(np.asarray(result.volume), expected)
                    self.assertEqual(int(result.stats["num_objects"]), num_objects)
                    self.assertEqual(int(result.stats["kept_objects"]), keep_n)
                    self.assertEqual(int(result.stats["kept_voxels"]), kept_voxels)
                    self.assertEqual(
                        int(result.stats["removed_voxels"]),
                        int(np.count_nonzero(source) - kept_voxels),
                    )
                    cuda_finalization._close_array(result.volume)

    def test_numpy_cuda_fake_failure_leaves_authority_untouched(self) -> None:
        source, _expected = self._physical_fixture()
        authority = source.copy()
        runtime = _FakeGpuRuntime(peer_access=True, fail_label_device=1)
        with tempfile.TemporaryDirectory() as temp_dir:
            result = self._run_physical_fake(runtime, source, temp_dir)
            self.assertIsNone(result)
            np.testing.assert_array_equal(source, authority)
            self.assertFalse(os.path.exists(os.path.join(
                temp_dir, "final_union_gpu_keep_candidate.u8.dat",
            )))
            self.assertEqual(sorted(runtime.pool_frees), [0, 1])

    def test_physical_equal_cutoff_tie_falls_back_without_mutating_authority(self) -> None:
        source = np.zeros((10, 7, 8), dtype=np.uint8)
        source[0:2, 1, 1] = np.uint8(1)
        source[8:10, 5, 6] = np.uint8(1)
        authority = source.copy()
        runtime = _FakeGpuRuntime(peer_access=True)
        with tempfile.TemporaryDirectory() as temp_dir:
            result = self._run_physical_fake(runtime, source, temp_dir)
            self.assertIsNone(result)
            np.testing.assert_array_equal(source, authority)
            self.assertFalse(os.path.exists(os.path.join(
                temp_dir, "final_union_gpu_keep_candidate.u8.dat",
            )))

    def test_disabled_physical_entry_does_not_import_cuda(self) -> None:
        mask = np.zeros((2, 3, 4), dtype=np.uint8)
        with mock.patch.dict(
            os.environ,
            {
                "YOLO_TTA_V1803_GPU_RESIDENT_TAIL": "0",
                "YOLO_TTA_V1803_GPU_RESIDENT_TAIL_REQUIRED": "0",
            },
        ):
            self.assertIsNone(
                cuda_finalization.try_apply_keep_largest_objects_multi_gpu(
                    mask, 1, os.getcwd(),
                )
            )

    def test_required_gate_cannot_silently_run_with_tail_disabled(self) -> None:
        mask = np.zeros((2, 3, 4), dtype=np.uint8)
        with mock.patch.dict(
            os.environ,
            {
                "YOLO_TTA_V1803_GPU_RESIDENT_TAIL": "0",
                "YOLO_TTA_V1803_GPU_RESIDENT_TAIL_REQUIRED": "1",
            },
        ):
            with self.assertRaisesRegex(RuntimeError, "REQUIRED=1 requires"):
                cuda_finalization.try_apply_keep_largest_objects_multi_gpu(
                    mask, 1, os.getcwd(),
                )


if __name__ == "__main__":
    unittest.main()
