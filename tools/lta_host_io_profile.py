"""Measure full-depth LTA union I/O with real masks and bounded file-backed storage.

No model or CUDA work is performed. The source masks are replayed into a zero
union at an explicit frame offset. Full-depth and active-span representations
use the same production hashing and coordinator reduction functions.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
import threading
import time
from types import SimpleNamespace
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import psutil
from XTA import lta_execution as execution, lta_worker_adapter as worker
from XTA.lta_outputs import write_json_atomically
from XTA.lta_union_artifacts import LtaUnionWriter, logical_union_sha256, reduce_union_artifact_into_view


class MemoryObserver:
    def __init__(self):
        self.process = psutil.Process()
        self.samples = []
        self.stage = "idle"
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        while not self.stop_event.is_set():
            self.samples.append({"time_unix": time.time(), "stage": self.stage,
                                 "rss_bytes": self.process.memory_info().rss,
                                 "available_bytes": psutil.virtual_memory().available})
            self.stop_event.wait(0.1)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *args):
        self.stop_event.set()
        self.thread.join(timeout=1)


def _close(array):
    mapped = getattr(array, "_mmap", None)
    if mapped is not None and not mapped.closed:
        mapped.close()


def run_trial(root, mode, *, depth, active_start, active, tile, observer):
    root.mkdir(parents=True, exist_ok=False)
    chain_path = root / "chain.uint8.raw"
    view_path = root / "view.uint8.raw"
    frames = depth if mode == "full_depth" else active.shape[0]
    chain_start = 0 if mode == "full_depth" else active_start
    local_start = active_start - chain_start
    stage_times = {}
    row = {"mode": mode, "chain_frames": frames,
           "active_frames": active.shape[0], "chain_bytes": frames * tile["size"]**2,
           "view_bytes": depth * tile["source_height"] * tile["source_width"],
           "stage_seconds": stage_times}
    def timed(name, function):
        observer.stage = root.name + ":" + name
        started = time.perf_counter()
        value = function()
        stage_times[name] = time.perf_counter() - started
        return value
    chain = view = None
    try:
        chain = timed("worker_union_create", lambda: np.memmap(
            chain_path, dtype=np.uint8, mode="w+", shape=(frames,tile["size"],tile["size"])))
        timed("write_active_masks", lambda: chain.__setitem__(slice(local_start,local_start+active.shape[0]),active))
        timed("worker_union_flush", chain.flush)
        foreground = timed("worker_foreground_scan", lambda: int(np.count_nonzero(chain)))
        assert foreground == int(np.count_nonzero(active))
        timed("worker_mapping_close", lambda: _close(chain))
        digest = timed("worker_union_sha256", lambda: worker._sha256_file(chain_path))
        view_plan = SimpleNamespace(frame_count=depth, frame_height=tile["source_height"], frame_width=tile["source_width"])
        view = timed("coordinator_view_create", lambda: execution._allocate_view_union(view_plan,view_path))
        manifest = {"union":{"path":str(chain_path),"sha256":digest,
                             "shape":[frames,tile["size"],tile["size"]]},
                    "tile":tile,"output_frame_range":[chain_start,chain_start+frames],"relays":[]}
        original_hash = execution._sha256_file
        original_reduce = execution.union_tile_chunk_into_view
        def checked_hash(path):
            return timed("coordinator_union_verify_sha256",lambda: original_hash(path))
        def reduce(*args,**kwargs):
            return timed("coordinator_dense_or",lambda: original_reduce(*args,**kwargs))
        with (mock.patch.object(execution,"_sha256_file",checked_hash),
              mock.patch.object(execution,"union_tile_chunk_into_view",reduce)):
            timed("coordinator_consume_total",lambda:execution._consume_chain_manifest(manifest,view_union=view))
        y,x,size=tile["top"],tile["left"],tile["size"]
        assert np.array_equal(view[active_start:active_start+active.shape[0],y:y+size,x:x+size],active)
        for frame in {0,depth-1,active_start-1,active_start+active.shape[0]}:
            if 0 <= frame < depth and not active_start <= frame < active_start+active.shape[0]:
                assert not bool(np.any(view[frame,y:y+size,x:x+size]))
        row["foreground_pixels"]=foreground
        row["active_mask_sha256"]=hashlib.sha256(memoryview(np.ascontiguousarray(active))).hexdigest()
        row["correctness"]="active masks exact; source full scan matches foreground count; off-span sentinel planes zero"
        timed("coordinator_view_flush",view.flush)
        timed("coordinator_view_close",lambda:_close(view))
        timed("worker_union_unlink",lambda:execution._unlink_consumed_temp_artifacts((chain_path,),temp_root=root))
        timed("view_unlink",lambda:execution._unlink_consumed_temp_artifacts((view_path,),temp_root=root))
        row["status"]="complete"
    finally:
        for array in (chain,view):
            if array is not None: _close(array)
        for path in (chain_path,view_path):
            if path.exists():
                execution._unlink_consumed_temp_artifacts((path,),temp_root=root)
        row["raw_files_removed"]=not chain_path.exists() and not view_path.exists()
        write_json_atomically(root/"trial.json",row)
    return row


class _ActiveFrameView:
    """Actual native row strides, with only the known active frames backed."""
    ndim = 3
    dtype = np.dtype(np.uint8)

    def __init__(self, backing, *, depth, active_start):
        self.backing = backing
        self.shape = (depth, *backing.shape[1:])
        self.flags = backing.flags
        self.active_start = active_start
        self.touched = []

    def __getitem__(self, key):
        frame, ys, xs = key
        if not isinstance(frame, int) or not self.active_start <= frame < self.active_start + self.backing.shape[0]:
            raise AssertionError("sparse consumer touched an omitted logical frame")
        self.touched.append((frame,ys.start,ys.stop,xs.start,xs.stop))
        return self.backing[frame-self.active_start,ys,xs]


def run_sparse_trial(root, *, depth, active_start, active, tile, observer):
    root.mkdir(parents=True,exist_ok=False)
    times = {}
    def timed(name, function):
        observer.stage = root.name + ":" + name
        started = time.perf_counter()
        value = function()
        times[name] = time.perf_counter() - started
        return value
    path=root/'union.rowpack.bin'
    backing_path=root/'active-view.raw'
    writer=LtaUnionWriter(path,shape=(depth,tile['size'],tile['size']),frame_start=0)
    backing=None
    try:
        timed('worker_sparse_append',lambda:writer.append_chunk(active_start,active))
        descriptor=timed('worker_sparse_finish',writer.finish)
        write_json_atomically(root/'union.json',descriptor)
        backing=timed('active_view_create',lambda:np.memmap(backing_path,dtype=np.uint8,mode='w+',
                       shape=(active.shape[0],tile['source_height'],tile['source_width'])))
        view=_ActiveFrameView(backing,depth=depth,active_start=active_start)
        x,y,size=tile['left'],tile['top'],tile['size']
        consumed=timed('coordinator_consume_total',lambda:reduce_union_artifact_into_view(
            descriptor,view_union=view,tile_xyxy=(x,y,x+size,y+size),frame_start=0,frame_stop=depth))
        assert np.array_equal(backing[:,y:y+size,x:x+size],active)
        assert int(np.count_nonzero(backing))==int(np.count_nonzero(active))
        assert {item[0] for item in view.touched}==set(range(active_start,active_start+active.shape[0]))
        timed('active_view_flush',backing.flush)
        touched_pixels=sum((y1-y0)*(x1-x0) for _frame,y0,y1,x0,x1 in view.touched)
        row={'status':'complete','mode':'sparse_full_logical_depth','stage_seconds':times,
             'logical_shape':[depth,tile['size'],tile['size']],
             'logical_bytes':depth*size*size,'stored_bytes':descriptor['size_bytes'],
             'storage_reduction_ratio':depth*size*size/max(1,descriptor['size_bytes']),
             'indexed_frames':len(descriptor['frames']),'foreground_pixels':descriptor['foreground_pixels'],
             'coordinator_touched_pixels':touched_pixels,'destination_backing_bytes':backing.nbytes,
             'full_depth_dense_allocation':False,'consumer_receipt':consumed,
             'active_mask_sha256':hashlib.sha256(memoryview(np.ascontiguousarray(active))).hexdigest(),
             'logical_hash_note':'logical hash intentionally excluded from production/timed path',
             'correctness':'every active pixel exact; total foreground exact; omitted logical frames never touched'}
        write_json_atomically(root/'trial.json',row)
        return row
    finally:
        writer.abort()
        if backing is not None: _close(backing)
        if backing_path.exists(): execution._unlink_consumed_temp_artifacts((backing_path,),temp_root=root)


def main():
    parser=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    parser.add_argument("--source-manifest",type=Path,required=True)
    parser.add_argument("--volume-depth",type=int,required=True)
    parser.add_argument("--active-start",type=int,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--repeats",type=int,default=2)
    parser.add_argument("--sparse-only",action="store_true",
                        help="Use full logical depth with <=30 backed destination frames; no full-depth dense allocation")
    args=parser.parse_args()
    source=json.loads(args.source_manifest.read_text())
    tile=dict(source["tile"])
    shape=tuple(source["union"]["shape"])
    if args.repeats<1 or not 0<=args.active_start<args.active_start+shape[0]<=args.volume_depth:
        parser.error("invalid frame extent or repeat count")
    output=args.output.resolve()
    if output.exists() and any(output.iterdir()): parser.error("output must be new or empty")
    output.mkdir(parents=True,exist_ok=True)
    live_bytes=args.volume_depth*(tile["source_width"]*tile["source_height"]+tile["size"]**2)
    if live_bytes>24*2**30 or shutil.disk_usage(output).free<live_bytes+15*2**30:
        raise RuntimeError("host I/O test exceeds 24GiB logical live-storage cap or 15GiB free-space reserve")
    if psutil.virtual_memory().available<12*2**30:
        raise RuntimeError("host I/O test requires at least12GiB currently available physical memory")
    source_path=Path(source["union"]["path"])
    if worker._sha256_file(source_path)!=source["union"]["sha256"]:
        raise RuntimeError("retained source union changed")
    active=np.fromfile(source_path,dtype=np.uint8).reshape(shape)
    record={"status":"running","source_manifest":str(args.source_manifest.resolve()),
            "source_union_sha256":source["union"]["sha256"],"volume_shape":[args.volume_depth,tile["source_height"],tile["source_width"]],
            "active_frame_range":[args.active_start,args.active_start+shape[0]],"tile":tile,
            "live_logical_bytes_cap":live_bytes,"data_amplification":args.volume_depth/shape[0],
            "source_code_sha256":{str(path.relative_to(ROOT)):worker._sha256_file(path) for path in
                                    (Path(__file__),ROOT/'XTA'/'lta_execution.py',ROOT/'XTA'/'lta_worker_adapter.py',ROOT/'XTA'/'lta_rendering.py')},
            "trials":[],"measurement_limits":"file-backed local host path; ordinary OS cache; no model or GPU work; view-create/flush costs shown separately from per-chain consumption"}
    with MemoryObserver() as observer:
        try:
            for repeat in range(args.repeats):
                if args.sparse_only:
                    row=run_sparse_trial(output/f"{repeat:02d}-sparse",depth=args.volume_depth,
                                         active_start=args.active_start,active=active,tile=tile,observer=observer)
                    record['trials'].append(row)
                    write_json_atomically(output/'summary.json',record)
                    print(json.dumps(row,sort_keys=True),flush=True)
                    continue
                for mode in (("active_span","full_depth") if repeat%2==0 else ("full_depth","active_span")):
                    row=run_trial(output/f"{repeat:02d}-{mode}",mode,depth=args.volume_depth,
                                  active_start=args.active_start,active=active,tile=tile,observer=observer)
                    record["trials"].append(row)
                    write_json_atomically(output/'summary.json',record)
                    print(json.dumps(row,sort_keys=True),flush=True)
            record["status"]="complete"
        finally:
            record["peak_rss_bytes"]=max((s["rss_bytes"] for s in observer.samples),default=0)
            record["minimum_available_bytes"]=min((s["available_bytes"] for s in observer.samples),default=0)
            write_json_atomically(output/'memory_samples.json',observer.samples)
            write_json_atomically(output/'summary.json',record)


if __name__=="__main__": main()
