"""Spawn four logical LTA workers over a full-depth sparse system-validation fixture.

Only SAM predictions are synthetic. Real rendering, mask wrapping, window DAG,
seeds, relay aggregation, sparse transport, and coordinator consumption run.
Unpopulated source frames are synthetic zero padding, not labeled source background.
This is not GPU throughput or model-quality evidence.
"""
from __future__ import annotations

import argparse
import ctypes
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace
from unittest import mock

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))

import numpy as np
from XTA import lta_execution as execution, lta_propagation as propagation
from XTA.lta_outputs import write_json_atomically
from XTA.lta_propagation import LtaMaskSeed, read_seed_artifact
from XTA.lta_rendering import LtaPhysicalViewCacheRef
from XTA.lta_runtime import LtaTileGridPlan
from XTA.lta_sam import SamFramePrediction
from XTA.lta_scheduler import LtaViewAffinityScheduler
from XTA.lta_telemetry import LtaExecutionTrace, lta_source_fingerprint
from XTA.lta_tile_tracking import LtaLineageId
from XTA.lta_tiles import plan_tile_grid
from XTA.lta_worker_adapter import execute_worker_task
from XTA.lta_workers import LtaWorkerInit,LtaWorkerPool


def allocated_bytes(path):
    if os.name!='nt': return Path(path).stat().st_blocks*512
    kernel=ctypes.WinDLL('kernel32',use_last_error=True)
    call=kernel.GetCompressedFileSizeW
    call.argtypes=(ctypes.c_wchar_p,ctypes.POINTER(ctypes.c_ulong)); call.restype=ctypes.c_ulong
    high=ctypes.c_ulong()
    low=call(str(path),ctypes.byref(high))
    if low==0xffffffff and ctypes.get_last_error(): raise ctypes.WinError(ctypes.get_last_error())
    return (high.value<<32)|low


def sparse_file(path,size):
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    with Path(path).open('xb'): pass
    if os.name=='nt':
        subprocess.run(['fsutil','sparse','setflag',str(path)],capture_output=True,check=True)
        if not (Path(path).stat().st_file_attributes & 0x200):
            raise RuntimeError('NTFS did not mark the fixture file sparse')
    with Path(path).open('r+b',buffering=0) as stream:
        if os.name=='nt':
            # CRT truncate zero-fills an extension even when the file is sparse.
            # SetEndOfFile preserves an unwritten NTFS hole instead.
            import msvcrt
            kernel=ctypes.WinDLL('kernel32',use_last_error=True)
            move=kernel.SetFilePointerEx
            move.argtypes=(ctypes.c_void_p,ctypes.c_longlong,ctypes.c_void_p,ctypes.c_ulong)
            move.restype=ctypes.c_int
            end=kernel.SetEndOfFile
            end.argtypes=(ctypes.c_void_p,); end.restype=ctypes.c_int
            handle=ctypes.c_void_p(msvcrt.get_osfhandle(stream.fileno()))
            if not move(handle,size,None,0) or not end(handle):
                raise ctypes.WinError(ctypes.get_last_error())
        else:
            stream.truncate(size)
    if allocated_bytes(path)>1024**2: raise RuntimeError('unwritten sparse fixture allocated dense storage')


def build_cpu_context(config):
    trace=LtaExecutionTrace(config['trace_dir'],'model_cpu')
    return SimpleNamespace(predictor=object(),profile={'name':'cpu_protocol_fixture'},
                           sam_runtime={'distribution_version':'synthetic_SAM_not_hardware'},
                           constrained_batches=None,config=dict(config),trace=trace,current_work=None)


def close_cpu_context(context): context.trace.close()


def execute_cpu_task(context,kind,payload):
    original=propagation.run_mask_injected_session
    context.current_work=payload['work_id']
    def sam_adapter(_measured,_raw,**kwargs):
        session,prompt=kwargs['session'],kwargs['prompt_frame']
        direction=kwargs['propagation_direction']
        frames=((*range(prompt,session.frame_stop),*range(prompt-1,session.frame_start-1,-1))
                if direction=='both' else range(prompt,session.frame_stop)
                if direction=='forward' else range(prompt,session.frame_start-1,-1))
        with context.trace.phase('synthetic_sam',work_id=context.current_work,
                                 device_id=int(os.environ['LTA_EXECUTION_DEVICE_ID']),direction=direction):
            time.sleep(float(context.config['model_delay']))
            for frame in frames:
                for object_id,seed in enumerate(kwargs['object_masks']):
                    active=context.config['active_start']<=frame<context.config['active_stop']
                    mask=np.asarray(seed).copy() if active else np.zeros_like(seed)
                    kwargs['prediction_callback'](SamFramePrediction(session.sequence_id,session.session_index,
                                                    frame,object_id,1.0,mask,0.9))
        return {'propagation':(),'seed_roundtrip_policy':'overlap-aware','seed_roundtrip_passed':True,
                'anchor_integrity_passed':True}
    def run(measured,raw,**kwargs): return original(measured,raw,adapter=sam_adapter,**kwargs)
    with mock.patch.object(propagation,'run_mask_injected_session',run):
        return execute_worker_task(context,kind,payload)


def run_case(root,*,windowed,cache,view,seed_mask,anchor,active_start,active_stop,model_delay):
    root.mkdir(parents=True,exist_ok=False)
    trace=LtaExecutionTrace(root/'traces','coordinator')
    grid=view.tile_grids[0]
    seeds=tuple(LtaMaskSeed(LtaLineageId(view.volume_id,'transverse',view.runtime_view_id,
                        f'pressure-lineage-{index}',grid.config_id),anchor,index,seed_mask,
                        visited_tile_indices=(0,),source_receipt={'fixture':'retained mask shape; synthetic placement/predictions'})
                for index in range(4))
    with mock.patch.object(execution,'_seeds_for_tile',side_effect=lambda *args,**kwargs:{anchor:seeds} if args[4]==0 else {}):
        chains,inventory=execution._plan_initial_chains(SimpleNamespace(volume_id=view.volume_id),(),view,cache,
                            temp_root=root,conf=0.15,empty_frame_limit=30)
    tasks=execution._plan_window_tasks(chains) if windowed else chains
    scheduler=LtaViewAffinityScheduler((task.work for task in tasks),(0,1,2,3),helper_queue_order='head',max_relay_generation=20)
    revisions={}
    execution._prime_authoritative_relay_destinations(scheduler,tasks[0].work.view,inventory,revisions)
    union_path=root/'view.sparse.raw'
    shape=(view.frame_count,view.frame_height,view.frame_width)
    sparse_file(union_path,int(np.prod(shape)))
    union=np.memmap(union_path,dtype=np.uint8,mode='r+',shape=shape)
    pool=None
    summary={'status':'running','windowed':windowed,'initial_logical_chains':len(chains),
             'initial_dispatch_units':len(tasks),'source_fingerprint':lta_source_fingerprint()}
    started=time.perf_counter()
    try:
        pool=LtaWorkerPool((0,1,2,3),LtaWorkerInit(adapter_module='tools.lta_host_pipeline_smoke',
                    adapter_factory='build_cpu_context',adapter_execute='execute_cpu_task',adapter_shutdown='close_cpu_context',
                    adapter_config={'trace_dir':str(root/'traces'),'active_start':active_start,'active_stop':active_stop,
                                    'model_delay':model_delay}),startup_timeout=60)
        result=execution._drive_workers_to_fixed_point(scheduler=scheduler,pool=pool,initial=tasks,
                    view_plan=view,cache_ref=cache,view_union=union,relay_mask_revisions=revisions,
                    temp_root=root,conf=0.15,empty_frame_limit=30,worker_task_timeout=120,trace=trace)
        active=np.asarray(union[active_start:active_stop])
        summary.update(status='complete',wall_seconds=time.perf_counter()-started,generation=result[0],
                       worker_pids=result[1],dispatches=result[2],worker_audit=result[3],
                       final_queue_counts=scheduler.queue_counts(),
                       active_native_sha256=hashlib.sha256(memoryview(active)).hexdigest(),
                       active_foreground_pixels=int(np.count_nonzero(active)),
                       native_view_logical_bytes=union_path.stat().st_size,
                       native_view_allocated_bytes=allocated_bytes(union_path))
        from XTA.lta_postprocessing import fill_binary_mask_holes_2d
        expected=fill_binary_mask_holes_2d(seed_mask)
        assert np.array_equal(active[:,:grid.tile_size,:grid.tile_size],np.broadcast_to(expected,(active_stop-active_start,*expected.shape)))
        assert summary['active_foreground_pixels']==int(np.count_nonzero(expected))*(active_stop-active_start)
        assert summary['native_view_allocated_bytes']<1024**3
        assert all(value==0 for value in summary['final_queue_counts'].values())
    except BaseException as exc:
        summary.update(status='failed',error={'type':type(exc).__name__,'message':str(exc)})
        raise
    finally:
        if pool is not None: summary['forced_shutdown_devices']=list(pool.shutdown(timeout=30))
        union.flush()
        summary['native_view_allocated_bytes_after_flush']=allocated_bytes(union_path)
        summary['native_view_is_sparse']=(bool(union_path.stat().st_file_attributes & 0x200)
                                          if os.name=='nt' else allocated_bytes(union_path)<union_path.stat().st_size)
        union._mmap.close()
        execution._unlink_consumed_temp_artifacts((union_path,),temp_root=root)
        summary['native_view_removed']=not union_path.exists()
        trace.close()
        write_json_atomically(root/'summary.json',summary)
    return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    parser.add_argument('--source-cache',type=Path,required=True)
    parser.add_argument('--source-cache-shape',type=int,nargs=3,required=True)
    parser.add_argument('--seed',type=Path,required=True)
    parser.add_argument('--volume-depth',type=int,required=True)
    parser.add_argument('--active-start',type=int,required=True)
    parser.add_argument('--anchor-frame',type=int,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--model-delay',type=float,default=0.05)
    args=parser.parse_args()
    count,height,width=args.source_cache_shape
    stop=args.active_start+count
    if not 0<=args.active_start<=args.anchor_frame<stop<=args.volume_depth or not 0<=args.model_delay<=2:
        parser.error('invalid frame range or bounded model delay')
    root=args.output.resolve()
    if root.exists() and any(root.iterdir()): parser.error('output must be new or empty')
    root.mkdir(parents=True,exist_ok=True)
    # These are logical CPU worker identities, not claims of four installed GPUs.
    os.environ.pop('CUDA_VISIBLE_DEVICES',None)
    shape=(args.volume_depth,height,width)
    cache_path=root/'source.sparse.gray8.raw'
    sparse_file(cache_path,int(np.prod(shape)))
    source=np.memmap(args.source_cache,dtype=np.uint8,mode='r',shape=(count,height,width))
    cache_array=np.memmap(cache_path,dtype=np.uint8,mode='r+',shape=shape)
    cache_array[args.active_start:stop]=source
    cache_array.flush(); cache_array._mmap.close(); source._mmap.close()
    if allocated_bytes(cache_path)>1024**3: raise RuntimeError('sparse source unexpectedly allocated dense storage')
    stat=cache_path.stat()
    identity=hashlib.sha256(json.dumps({'shape':shape,'active_start':args.active_start,
                'source':str(args.source_cache.resolve()),'padding':'synthetic_zero'},sort_keys=True).encode()).hexdigest()
    cache=LtaPhysicalViewCacheRef(cache_path,shape,'uint8','transverse',identity,stat.st_size,stat.st_mtime_ns)
    seed=read_seed_artifact(args.seed)[0]
    size=seed.mask.shape[0]
    grid=LtaTileGridPlan(f's{size}_st{3*size//4}',size,3*size//4,
                         plan_tile_grid(source_width=width,source_height=height,tile_size=size,tile_stride=3*size//4))
    view=SimpleNamespace(volume_id='full-depth-system-fixture',physical_view_id='transverse',
                         runtime_view_id='transverse__tta_a0',frame_count=args.volume_depth,
                         frame_height=height,frame_width=width,tile_grids=(grid,))
    report={'status':'running','fixture':'30real source frames, synthetic zero padding and mocked SAM; system validation only',
            'source_shape':shape,'active_frame_range':[args.active_start,stop],
            'source_logical_bytes':stat.st_size,'source_allocated_bytes':allocated_bytes(cache_path)}
    try:
        for name,windowed in [('whole_chain',False),('window_dag',True)]:
            report[name]=run_case(root/name,windowed=windowed,cache=cache,view=view,seed_mask=seed.mask,
                           anchor=args.anchor_frame,active_start=args.active_start,active_stop=stop,model_delay=args.model_delay)
        assert report['whole_chain']['active_native_sha256']==report['window_dag']['active_native_sha256']
        report['status']='passed'
        report['exact_native_mask_parity']=True
    except BaseException as exc:
        report['status']='failed'
        report['error']={'type':type(exc).__name__,'message':str(exc)}
        raise
    finally:
        execution._unlink_consumed_temp_artifacts((cache_path,),temp_root=root)
        report['source_sparse_file_removed']=not cache_path.exists()
        write_json_atomically(root/'summary.json',report)
    print(json.dumps({'status':report['status'],'output':str(root)},sort_keys=True),flush=True)


if __name__=='__main__': main()
