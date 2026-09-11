"""Compare real single-device LTA chain and window-DAG execution functionally.

Masks, per-lineage prediction scores, and relay seed identities are checked.
This diagnostic makes no speed claim and performs no benchmark heatsoak.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest import mock

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))

import numpy as np
from XTA import lta_execution as execution, lta_propagation as propagation
from XTA.lta_outputs import write_json_atomically
from XTA.lta_propagation import read_seed_artifact
from XTA.lta_rendering import reference_existing_physical_view_cache
from XTA.lta_runtime import LtaTileGridPlan
from XTA.lta_scheduler import LtaViewAffinityScheduler
from XTA.lta_telemetry import LtaExecutionTrace,lta_source_fingerprint
from XTA.lta_tiles import plan_tile_grid
from XTA.lta_worker_adapter import build_worker_predictor,close_worker_predictor,execute_worker_task
from XTA.lta_workers import LtaWorkerInit,LtaWorkerPool


def build_context(config):
    context=build_worker_predictor(config)
    context.score_root=Path(config['score_root'])
    context.score_root.mkdir(parents=True,exist_ok=True)
    return context


def execute_traced(context,kind,payload):
    original=propagation.run_mask_injected_session
    records=[]
    def traced(measured,raw,**kwargs):
        request=kwargs['request']; callback=kwargs['prediction_callback']
        def observe(item):
            prediction=item.prediction
            array=np.ascontiguousarray(prediction.binary_mask,dtype=np.uint8)
            records.append({'lineage':item.lineage.token,'frame':int(prediction.frame_index),
                            'session_range':[request.session.frame_start,request.session.frame_stop],
                            'direction':request.direction,'mask_sha256':hashlib.sha256(memoryview(array)).hexdigest(),
                            'initial_detection_score':prediction.initial_detection_score,
                            'frame_tracker_score':prediction.frame_tracker_score})
            callback(item)
        kwargs['prediction_callback']=observe
        return original(measured,raw,**kwargs)
    with mock.patch.object(propagation,'run_mask_injected_session',traced):
        result=execute_worker_task(context,kind,payload)
    name=hashlib.sha256(payload['work_id'].encode()).hexdigest()[:24]
    path=context.score_root/(payload['qualification_role']+'-'+name+'.json')
    write_json_atomically(path,records)
    return result


def run_case(root,*,pool,cache,view,seeds,windowed,role,device):
    root.mkdir(parents=True,exist_ok=False)
    by_frame={seeds[0].frame_index:seeds}
    with mock.patch.object(execution,'_seeds_for_tile',side_effect=lambda *args,**kwargs:by_frame if args[4]==0 else {}):
        chains,inventory=execution._plan_initial_chains(SimpleNamespace(volume_id=view.volume_id),(),view,cache,
                                        temp_root=root,conf=0.15,empty_frame_limit=30)
    tasks=execution._plan_window_tasks(chains) if windowed else chains
    # Preserve the role through the driver's dynamically created relay payloads.
    original_submit=pool.submit
    def submit(task,*,execution_device_id):
        from dataclasses import replace
        task=replace(task,payload={**dict(task.payload),'qualification_role':role})
        return original_submit(task,execution_device_id=execution_device_id)
    scheduler=LtaViewAffinityScheduler((task.work for task in tasks),(device,),helper_queue_order='head',max_relay_generation=20)
    revisions={}
    execution._prime_authoritative_relay_destinations(scheduler,tasks[0].work.view,inventory,revisions)
    union=np.zeros(cache.shape,dtype=np.uint8)
    trace=LtaExecutionTrace(root/'traces','coordinator')
    relay_records=[]
    original_plan=execution._plan_relay_generation
    def plan_relays(records,**kwargs):
        for record in records:
            seeds_in=read_seed_artifact(record['seed_artifact_path'],expected_sha256=record['seed_artifact_sha256'])
            relay_records.append({'generation':kwargs['generation'],
                'destination':record['destination_tile_index'],'frame':record['frame_index'],
                'direction':record['temporal_direction'],'artifact_sha256':record['seed_artifact_sha256'],
                'seeds':[{'lineage':seed.lineage.token,'frame':seed.frame_index,
                          'mask_sha256':hashlib.sha256(np.ascontiguousarray(seed.mask,dtype=np.uint8).tobytes()).hexdigest(),
                          'probability':seed.tracker_probability} for seed in seeds_in]})
        return original_plan(records,**kwargs)
    try:
        with (mock.patch.object(pool,'submit',submit),mock.patch.object(execution,'_plan_relay_generation',plan_relays)):
            result=execution._drive_workers_to_fixed_point(scheduler=scheduler,pool=pool,initial=tasks,
                        view_plan=view,cache_ref=cache,view_union=union,relay_mask_revisions=revisions,temp_root=root,
                        conf=0.15,empty_frame_limit=30,worker_task_timeout=600,trace=trace)
        summary={'status':'complete','source_fingerprint':lta_source_fingerprint(),'logical_union_sha256':hashlib.sha256(memoryview(union)).hexdigest(),
                 'foreground_pixels':int(np.count_nonzero(union)),'generation':result[0],
                 'dispatches':result[2],'worker_audit':result[3],'relays':relay_records,
                 'final_queue_counts':scheduler.queue_counts()}
        write_json_atomically(root/'summary.json',summary)
        return summary
    finally:
        trace.close()


def main():
    parser=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    parser.add_argument('--model',required=True)
    parser.add_argument('--fixtures',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--profile',choices=('auto','egpu','h100'),default='auto')
    parser.add_argument('--device',type=int,default=0)
    args=parser.parse_args()
    root=args.output.resolve()
    if root.exists() and any(root.iterdir()): parser.error('output must be new or empty')
    root.mkdir(parents=True,exist_ok=True)
    fixtures=json.loads(args.fixtures.read_text())
    report={'status':'running','fixture_results':{},'qualification':'single real GPU functional protocol parity; not a speed benchmark'}
    pool=LtaWorkerPool((args.device,),LtaWorkerInit(adapter_module='tools.lta_window_gpu_smoke',adapter_factory='build_context',
                 adapter_execute='execute_traced',adapter_shutdown='close_worker_predictor',adapter_config={
                     'model_path':args.model,'profile':args.profile,'conf':0.15,'score_root':str(root/'scores'),
                     'trace_dir':str(root/'worker-traces')}),startup_timeout=600)
    primary_error=None
    try:
        for fixture in fixtures:
            name=fixture['name']; seeds=read_seed_artifact(fixture['seed'])
            shape=tuple(fixture['cache_shape']); size=seeds[0].mask.shape[0]
            cache=reference_existing_physical_view_cache(Path(fixture['cache']),shape=shape,
                       physical_view_id=seeds[0].lineage.physical_view_id,source_identity='immutable-'+name)
            grid=LtaTileGridPlan(seeds[0].lineage.tile_config_id,size,3*size//4,
                       plan_tile_grid(source_width=shape[2],source_height=shape[1],tile_size=size,tile_stride=3*size//4))
            view=SimpleNamespace(volume_id=seeds[0].lineage.volume_id,physical_view_id=seeds[0].lineage.physical_view_id,
                   runtime_view_id=seeds[0].lineage.runtime_view_id,frame_count=shape[0],frame_height=shape[1],frame_width=shape[2],tile_grids=(grid,))
            values={}
            for role,windowed in [('whole',False),('windows',True)]:
                values[role]=run_case(root/(name+'-'+role),pool=pool,cache=cache,view=view,seeds=seeds,
                                      windowed=windowed,role=name+'-'+role,device=args.device)
            traces={role:sorted((item for path in (root/'scores').glob(name+'-'+role+'-*.json')
                                    for item in json.loads(path.read_text())),key=lambda row:json.dumps(row,sort_keys=True))
                    for role in ('whole','windows')}
            relay_sort=lambda value:sorted(value,key=lambda row:json.dumps(row,sort_keys=True))
            comparisons={'native_mask_exact':values['whole']['logical_union_sha256']==values['windows']['logical_union_sha256'],
                         'prediction_masks_scores_exact':traces['whole']==traces['windows'],
                         'relay_artifacts_exact':relay_sort(values['whole']['relays'])==relay_sort(values['windows']['relays']),
                         'prediction_counts':{role:len(rows) for role,rows in traces.items()}}
            report['fixture_results'][name]={'runs':values,'comparisons':comparisons}
            write_json_atomically(root/'summary.json',report)
            if not all(comparisons[key] for key in ('native_mask_exact','prediction_masks_scores_exact','relay_artifacts_exact')):
                raise RuntimeError(f'legacy/window functional parity failed: {name}: {comparisons}')
        report['status']='passed'
    except BaseException as exc:
        primary_error=exc
        report.update(status='failed',error={'type':type(exc).__name__,'message':str(exc)})
        raise
    finally:
        try:
            report['forced_shutdown_devices']=list(pool.shutdown(timeout=30))
            if report['forced_shutdown_devices']:
                raise RuntimeError('functional comparison required forced worker shutdown')
        except BaseException as shutdown_error:
            report['status']='failed'; report['shutdown_error']=str(shutdown_error)
            if primary_error is None: raise
            if hasattr(primary_error,'add_note'): primary_error.add_note(str(shutdown_error))
        finally:
            write_json_atomically(root/'summary.json',report)
    print(json.dumps({'status':report['status'],'output':str(root)},sort_keys=True),flush=True)


if __name__=='__main__': main()
