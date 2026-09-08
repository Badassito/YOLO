from __future__ import annotations
import unittest
from unittest import mock
import os
import sys
import json
import gzip
import tempfile
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from contextlib import ExitStack
from concurrent.futures import ThreadPoolExecutor
from XTA import publication_memory as memory
from XTA import interpolation
from XTA.config import GIB
from XTA.interpolation import IncrementalRawBBoxMaskStoreWriter, RawBBoxMaskStore, INTERNAL_PACKED_CVOL_FORMAT
from XTA.runtime import release_memfd_owners_under


def _cache_export_volumes():
    a = np.zeros((5,17,43), np.uint8)
    pattern = (np.indices((5,13,33)).sum(axis=0) % 2).astype(np.uint8)
    a[:,2:15,4:37] = pattern
    b = np.zeros_like(a); b[:,2:15,4:37] = 1-pattern
    c = np.zeros_like(a); c[:,1:16,2:41] = 1
    return a,b,c


def _verify_concurrent_cached_exports(root, volumes):
    """Hold multiple cached layers through the real native/mirror NRRD writers."""
    from XTA import outputs
    refs = [interpolation.NrrdLayerRef(key=f'layer{i}', name=f'layer{i}', path=root/f'store{i}',
                shape=a.shape, storage_format=INTERNAL_PACKED_CVOL_FORMAT)
            for i,a in enumerate(volumes)]
    def export(i, tag):
        native = root/f'{tag}-{i}.nrrd'; mirror = root/f'{tag}-{i}-mirror.nrrd'
        outputs.write_layer_nrrd_with_low_quality_mirrors(refs[i], volumes[i].shape,
                native, [((2,7,11), mirror)], segment_name=f'layer{i}')
        def decode(path):
            _header, payload = path.read_bytes().split(b'\n\n', 1)
            return gzip.decompress(payload)
        result = decode(native), decode(mirror)
        np.testing.assert_array_equal(np.frombuffer(result[0], np.uint8).reshape(volumes[i].shape), volumes[i])
        return result
    def software_writer(fh, **kwargs):
        return outputs._MemberParallelGzipPayloadWriter(fh, codec_spec=('zlib',1,gzip.compress))
    with mock.patch.object(outputs, '_open_nrrd_payload_writer', side_effect=software_writer):
        # Independent uncached reads establish both native and mirror oracles.
        with mock.patch.object(outputs, '_open_nrrd_layer_ref',
                side_effect=lambda ref: RawBBoxMaskStore.open(ref.path, mmap_payload=True)):
            expected = [export(i, 'uncached') for i in range(len(refs))]
        with ExitStack() as stack:
            for ref in refs:
                reader = RawBBoxMaskStore.open(ref.path, cache_payload_in_ram=True)
                stack.callback(interpolation._release_raw_store_chunks_ram_cache, reader.chunks_path)
                stack.callback(reader.close)
            with ThreadPoolExecutor(max_workers=len(refs)) as pool:
                futures = [pool.submit(export, i, 'cached') for i in range(len(refs))]
                actual = [future.result() for future in futures]
            if actual != expected:
                raise AssertionError('Cached layer or low-quality mirror contains another layer payload')


def _verify_cached_exports_in_subprocess(root, *, emulate_memfd_resolution):
    # Aggregate discovery substitutes dependency stubs. Exercise the actual
    # OpenCV mirror path in a fresh interpreter, as the other numerical fixtures do.
    import subprocess
    program = '''
import sys
from pathlib import Path
from unittest import mock
from tests.test_publication_memory import _cache_export_volumes,_verify_concurrent_cached_exports
resolve=Path.resolve
def display(path,*args,**kwargs):
    if sys.argv[2]=='1' and path.name=='chunks.bin':
        return Path('/memfd:xta-packed-publication (deleted)')
    return resolve(path,*args,**kwargs)
with mock.patch.object(Path,'resolve',display):
    _verify_concurrent_cached_exports(Path(sys.argv[1]),_cache_export_volumes())
'''
    result = subprocess.run([sys.executable, '-c', program, str(root),
                             '1' if emulate_memfd_resolution else '0'],
                            text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=60)
    if result.returncode:
        raise AssertionError(result.stdout)


class PublicationMemoryPlanTests(unittest.TestCase):
    def setUp(self):
        def bbox(a):
            ys, xs = np.nonzero(a)
            return ((int(xs.min()), int(ys.min()), int(xs.max()-xs.min()+1), int(ys.max()-ys.min()+1))
                    if len(xs) else (0, 0, 0, 0))
        patcher = mock.patch('XTA.interpolation.cv2.boundingRect', side_effect=bbox)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_same_named_memfds_cannot_share_cached_layer_bytes(self):
        # Linux /proc/<pid>/fd/N resolves different, equally named memfds to the
        # same /memfd:name (deleted) display string. Emulate only that resolution;
        # use real files and mmap acquisition/refcounts on every platform.
        resolve = Path.resolve
        def memfd_display(path, *args, **kwargs):
            if path.name == 'chunks.bin':
                return Path('/memfd:xta-packed-publication (deleted)')
            return resolve(path, *args, **kwargs)
        for payloads in ((b'abcdefgh', b'ABCDEFGH'), (b'abcdefgh', b'0123456789abcdef')):
            with self.subTest(lengths=tuple(map(len, payloads))), tempfile.TemporaryDirectory() as td:
                paths = [Path(td)/str(i)/'chunks.bin' for i in range(2)]
                for path, data in zip(paths, payloads):
                    path.parent.mkdir(); path.write_bytes(data)
                acquired = []
                with mock.patch.object(Path, 'resolve', memfd_display):
                    try:
                        a, reused = interpolation._acquire_raw_store_chunks_ram_cache(paths[0])
                        acquired.append(paths[0]); self.assertFalse(reused)
                        b, reused = interpolation._acquire_raw_store_chunks_ram_cache(paths[1])
                        acquired.append(paths[1]); self.assertFalse(reused)
                        self.assertIsNot(a, b)
                        self.assertEqual(bytes(a), payloads[0])
                        self.assertEqual(bytes(b), payloads[1])
                        again, reused = interpolation._acquire_raw_store_chunks_ram_cache(paths[0])
                        acquired.append(paths[0]); self.assertTrue(reused)
                        self.assertIs(again, a)
                        # Rewriting one logical layer must not evict another layer.
                        interpolation._invalidate_raw_store_chunks_ram_cache(paths[0])
                        self.assertEqual(bytes(b), payloads[1])
                    finally:
                        for path in reversed(acquired):
                            interpolation._release_raw_store_chunks_ram_cache(path)

    def test_cached_payload_can_retire_after_proc_target_disappears(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td)/'chunks.bin'; path.write_bytes(b'owned bytes')
            with mock.patch.object(Path, 'resolve', return_value=Path('/memfd:shared (deleted)')):
                payload, _reused = interpolation._acquire_raw_store_chunks_ram_cache(path)
            try:
                with mock.patch.object(Path, 'resolve', side_effect=FileNotFoundError('owner retired')):
                    interpolation._release_raw_store_chunks_ram_cache(path)
                self.assertTrue(payload.closed)
            finally:
                # Also clean up the intentionally broken implementation in the
                # red regression, whose stale key otherwise keeps Windows mmap open.
                with mock.patch.object(Path, 'resolve', return_value=Path('/memfd:shared (deleted)')):
                    interpolation._invalidate_raw_store_chunks_ram_cache(path)

    def test_concurrent_cached_nrrds_with_same_memfd_display_name(self):
        volumes = _cache_export_volumes()
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            for i,a in enumerate(volumes):
                writer = IncrementalRawBBoxMaskStoreWriter(shape=a.shape, store_dir=root/f'store{i}',
                    format_name=INTERNAL_PACKED_CVOL_FORMAT, desc='cached-export fixture')
                try:
                    writer.consume(0,a); writer.finalize()
                except BaseException:
                    writer.discard(); raise
            # First two payloads have identical length but different pixel bytes.
            self.assertEqual((root/'store0/chunks.bin').stat().st_size, (root/'store1/chunks.bin').stat().st_size)
            _verify_cached_exports_in_subprocess(root, emulate_memfd_resolution=True)

    @unittest.skipUnless(hasattr(os, 'memfd_create'), 'Linux memfd unavailable')
    def test_multiple_parent_memfds_survive_producer_exit_and_cached_exports(self):
        import subprocess
        volumes = _cache_export_volumes()
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            tasks = [dict(projection_contract='radial_native_pull_v1', result_mode='d1_owner',
                          d1_store_dir=str(root/f'store{i}'), d1_output_shape=list(a.shape),
                          model_name='m', view=SimpleNamespace(name=f'radial-{i}'))
                     for i,a in enumerate(volumes)]
            with mock.patch.object(memory, 'publication_ram_headroom', return_value=100*GIB), \
                    mock.patch.object(memory, 'scratch_dir_is_memory_backed', return_value=False), \
                    mock.patch.object(memory, 'workspace_anon_cap_bytes', return_value=0), \
                    mock.patch.dict(os.environ, {'YOLO_TTA_MEMFD_WORKSPACES':'1','YOLO_TTA_PUBLICATION_RAM':'1',
                                                'YOLO_TTA_PACKED_OWNER_PUBLICATION':'1','YOLO_TTA_PUBLICATION_RAM_GIB':'0'}):
                memory.plan_native_publication_memory(tasks, keep_temp=False, worker_count=1,
                                                      publication_pending=1, unpack_bytes=1024)
            program = '''
import json,sys
from pathlib import Path
from tests.test_publication_memory import _cache_export_volumes
from XTA.interpolation import IncrementalRawBBoxMaskStoreWriter,INTERNAL_PACKED_CVOL_FORMAT
for task,a in zip(json.loads(sys.argv[1]), _cache_export_volumes()):
    w=IncrementalRawBBoxMaskStoreWriter(shape=a.shape,store_dir=Path(task['d1_store_dir']),format_name=INTERNAL_PACKED_CVOL_FORMAT,desc='child multi-layer',payload_backing=task['d1_memory_payload_path'])
    try:
        w.consume(0,a); assert w.finalize()['payload_backing']=='planned_memfd'
    except BaseException:
        w.discard(); raise
'''
            try:
                subprocess.run([sys.executable, '-c', program,
                    json.dumps([{k:v for k,v in t.items() if k!='view'} for t in tasks])], check=True)
                _verify_cached_exports_in_subprocess(root, emulate_memfd_resolution=False)
            finally:
                self.assertEqual(release_memfd_owners_under(root), len(volumes))

    def test_large_codec_windows_and_mirror_canvases_reduce_retention_budget(self):
        sink = SimpleNamespace(max_workers=12, output_shape=(1931,3064,3022),
                low_quality_specs=[SimpleNamespace(output_shape_t_y_x=(388,612,604))])
        small = memory.publication_output_reserve(sink, 512*1024**2, 16*1024**2)
        huge = memory.publication_output_reserve(sink, 16*GIB, 16*1024**2)
        self.assertGreater(huge, small)
        normal = memory.retained_payload_plan([sink.output_shape]*50, 950*GIB, 4, 12, 256*1024**2,
                                               output_reserve_bytes=small)
        limited = memory.retained_payload_plan([sink.output_shape]*50, 950*GIB, 4, 12, 256*1024**2,
                                                output_reserve_bytes=huge)
        self.assertGreater(normal['budget_bytes'], limited['budget_bytes'])

    def test_small_layers_cannot_exhaust_parent_file_descriptors(self):
        tasks = [dict(projection_contract='radial_native_pull_v1', result_mode='d1_owner',
                      d1_store_dir=f'not-created-store-{i}', d1_output_shape=[2,4,16],
                      model_name='m', view=SimpleNamespace(name=f'radial-{i}')) for i in range(10)]
        resource = SimpleNamespace(RLIMIT_NOFILE=7, getrlimit=lambda _which: (80,80))
        with mock.patch.dict(sys.modules, {'resource':resource}), \
                mock.patch.object(memory, 'memfd_workspace_enabled', return_value=True), \
                mock.patch.object(memory, 'scratch_dir_is_memory_backed', return_value=False), \
                mock.patch.object(memory, 'workspace_anon_cap_bytes', return_value=0), \
                mock.patch.object(memory, 'publication_ram_headroom', return_value=100*GIB), \
                mock.patch.object(memory.os, 'listdir', return_value=[]), \
                mock.patch.object(memory.os, 'memfd_create', create=True, side_effect=range(10,20)) as create, \
                mock.patch.object(memory, '_register_memfd_owner'), \
                mock.patch.dict(os.environ, {'YOLO_TTA_PUBLICATION_RAM':'1',
                                            'YOLO_TTA_PACKED_OWNER_PUBLICATION':'1','YOLO_TTA_PUBLICATION_RAM_GIB':'0'}):
            plan = memory.plan_native_publication_memory(tasks, keep_temp=False, worker_count=1,
                                                        publication_pending=1, unpack_bytes=1024)
        self.assertEqual(create.call_count, 8)
        self.assertEqual(sum('d1_memory_payload_path' in t for t in tasks), 8)
        self.assertEqual(plan['grants'][-2:], [0,0])
        self.assertEqual(plan['reserved_bytes'], sum(plan['grants']))

    def test_production_plan_charges_every_future_layer_and_reserves_tail(self):
        shape = (1931, 3064, 3022)
        plan = memory.retained_payload_plan([shape]*50, 950*GIB, 4, 12, 256*1024**2)
        self.assertEqual(len([v for v in plan['grants'] if v]), 50)
        self.assertEqual(sum(plan['grants']), plan['reserved_bytes'])
        self.assertLessEqual(plan['reserved_bytes'], plan['budget_bytes'])
        self.assertGreater(plan['reserve_bytes'], 200*GIB)
        self.assertLessEqual(plan['reserved_bytes'] + plan['reserve_bytes'], 950*GIB)

    def test_small_job_limit_and_explicit_cap_spill_instead_of_overcommitting(self):
        shapes = [(1931, 3064, 3022)]*500
        low = memory.retained_payload_plan(shapes, 100*GIB, 4, 12, 256*1024**2)
        self.assertEqual(sum(low['grants']), 0)
        bounded = memory.retained_payload_plan(shapes, 950*GIB, 4, 12, 256*1024**2, cap=8*GIB)
        self.assertLessEqual(sum(bounded['grants']), 8*GIB)
        self.assertTrue(any(bounded['grants']))
        self.assertTrue(any(v == 0 for v in bounded['grants']))

    def test_swap_is_never_counted_as_ram_headroom(self):
        with mock.patch.object(memory, '_read_meminfo_bytes', return_value={'MemAvailable': 10, 'SwapFree': 1000}), \
                mock.patch.object(memory, 'available_anon_work_bytes', return_value=1010):
            self.assertEqual(memory.publication_ram_headroom(), 10)
        with mock.patch.object(memory, '_read_meminfo_bytes', return_value={'MemAvailable': 1000}), \
                mock.patch.object(memory, 'available_anon_work_bytes', return_value=20):
            self.assertEqual(memory.publication_ram_headroom(), 20)

    def test_retained_scratch_never_uses_ephemeral_parent_descriptors(self):
        with mock.patch.object(memory.os, 'memfd_create', create=True) as create:
            self.assertIsNone(memory.plan_native_publication_memory([], keep_temp=True,
                              worker_count=4, publication_pending=12, unpack_bytes=256*1024**2))
            create.assert_not_called()

    def test_private_payload_spill_preserves_bytes_and_releases_original_backing(self):
        volume = np.random.default_rng(13).integers(0, 2, (5, 17, 43), dtype=np.uint8)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            backing = root/'parent-backed-payload'
            backing.touch()
            # A hardlink stands in for the Linux /proc symlink on Windows. Both
            # descriptors refer to the same payload and replacement preserves it.
            def link(path, target, **kwargs):
                os.link(target, path)
            original_open = os.open
            def share_delete_open(path, flags, mode=0o666):
                if os.name != 'nt':
                    return original_open(path, flags, mode)
                # Linux permits replacing open files. Give the Windows fixture
                # that same sharing behavior; production RAM backings are Linux.
                import _winapi
                import msvcrt
                disposition = 1 if flags & os.O_EXCL else (2 if flags & os.O_TRUNC else (4 if flags & os.O_CREAT else 3))
                handle = _winapi.CreateFile(str(path), 0xc0000000, 7, 0, disposition, 0, 0)
                return msvcrt.open_osfhandle(handle, os.O_RDWR | os.O_BINARY)
            with ExitStack() as stack:
                stack.enter_context(mock.patch.object(os, 'open', share_delete_open))
                stack.enter_context(mock.patch.object(Path, 'symlink_to', link))
                writer = IncrementalRawBBoxMaskStoreWriter(shape=volume.shape, store_dir=root/'store',
                    format_name=INTERNAL_PACKED_CVOL_FORMAT, desc='spill-test', payload_backing=str(backing))
                stack.callback(writer.discard)
                writer.consume(0, volume[:2])
                with mock.patch.object(writer, '_pwrite_all', side_effect=OSError('disk full')):
                    with self.assertRaisesRegex(OSError, 'disk full'):
                        writer.spill_payload_to_disk()
                self.assertTrue(writer._ram_payload)
                self.assertGreater(backing.stat().st_size, 0)
                writer.spill_payload_to_disk()
                self.assertEqual(backing.stat().st_size, 0)
                writer.consume(2, volume[2:])
                self.assertEqual(writer.finalize()['payload_backing'], 'disk')
                reader = RawBBoxMaskStore.open(writer.store_dir, mmap_payload=True)
                try:
                    np.testing.assert_array_equal(np.stack([reader.decode_slice(z) for z in range(5)]), volume)
                finally:
                    reader.close()

    @unittest.skipUnless(hasattr(os, 'memfd_create'), 'Linux memfd unavailable')
    def test_parent_descriptor_survives_spawned_producer_and_retires(self):
        import subprocess
        import sys
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            view = SimpleNamespace(name='radial-test')
            task = dict(projection_contract='radial_native_pull_v1', result_mode='d1_owner',
                        d1_store_dir=str(root/'store'), d1_output_shape=[3, 5, 43], model_name='m', view=view)
            with mock.patch.object(memory, 'publication_ram_headroom', return_value=100*GIB), \
                    mock.patch.object(memory, 'scratch_dir_is_memory_backed', return_value=False), \
                    mock.patch.object(memory, 'workspace_anon_cap_bytes', return_value=0), \
                    mock.patch.dict(os.environ, {'YOLO_TTA_MEMFD_WORKSPACES':'1', 'YOLO_TTA_PUBLICATION_RAM':'1',
                                                'YOLO_TTA_PACKED_OWNER_PUBLICATION':'1', 'YOLO_TTA_PUBLICATION_RAM_GIB':'0'}):
                memory.plan_native_publication_memory([task], keep_temp=False, worker_count=1,
                                                      publication_pending=1, unpack_bytes=1024)
            program = '''
import sys,json,numpy as np
from XTA.interpolation import IncrementalRawBBoxMaskStoreWriter,INTERNAL_PACKED_CVOL_FORMAT
from pathlib import Path
t=json.loads(sys.argv[1]); a=np.arange(3*5*43).reshape(3,5,43)%2
w=IncrementalRawBBoxMaskStoreWriter(shape=a.shape,store_dir=Path(t['d1_store_dir']),format_name=INTERNAL_PACKED_CVOL_FORMAT,desc='child',payload_backing=t['d1_memory_payload_path'])
w.consume(0,a); assert w.finalize()['payload_backing']=='planned_memfd'
'''
            try:
                subprocess.run([sys.executable, '-c', program, json.dumps({k: v for k, v in task.items() if k != 'view'})], check=True)
                reader = RawBBoxMaskStore.open(root/'store', mmap_payload=True)
                try:
                    np.testing.assert_array_equal(np.stack([reader.decode_slice(z) for z in range(3)]),
                                                  np.arange(3*5*43).reshape(3,5,43)%2)
                finally:
                    reader.close()
            finally:
                self.assertEqual(release_memfd_owners_under(root), 1)


if __name__ == '__main__':
    unittest.main()
