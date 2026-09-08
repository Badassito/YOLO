"""Qualify native owner chunks on production geometry; no model timing claims."""
from dataclasses import replace
import argparse
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import time

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from XTA import geometry
from XTA.cylindrical_owner import RadialOwner, RADIAL_OWNER_CONTRACT
from XTA.cuda_d1 import _d1_finalize_bitset_layer
from tools.benchmark_radial_cuda_projection import WORKING_SHAPE, OUTPUT_SHAPE
from tools.benchmark_radial_graph_dispatch import fill_ellipsoid_source
from tools.benchmark_radial_cuda_sink import decode_store
from tools.benchmark_radial_setup import heatsoak


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--chunk',type=int,default=32)
    parser.add_argument('--heat-seconds',type=float,default=60.)
    args=parser.parse_args()
    if args.chunk<1 or args.heat_seconds<0:
        parser.error('Chunk must be positive and heat time nonnegative')
    root=args.output_dir.resolve();root.mkdir(parents=True,exist_ok=True)
    import cupy as cp
    view=geometry.get_view_infos(*WORKING_SHAPE,cartesian_views=(),radial_views=('transverse',),radial_patch_size=3072)[0]
    view=replace(view,radial_tilted_source=True,tilt_direction='horizontal',tilt_angle_deg=-30.)
    report={'source_shape':(view.num_slices,3072,3072),'output_shape':OUTPUT_SHAPE,'chunks':[],
        'limits':'Synthetic spatial ellipsoid, no model inference or four-GPU scheduling. Source chunk '
                 'uploads are fixture setup and timed separately; production receives already-resident native masks.'}
    from XTA.cylindrical_owner import _OWNER_KERNEL_SOURCE
    report['kernel_sha256']=hashlib.sha256(_OWNER_KERNEL_SOURCE.encode()).hexdigest()
    report['heatsoak_seconds']=heatsoak(args.heat_seconds,0)
    with tempfile.TemporaryDirectory(prefix='owner-fixture-',dir=root) as directory:
        work=Path(directory).resolve()
        if work.parent!=root:
            raise RuntimeError('Unexpected benchmark temporary directory')
        source=np.memmap(work/'source.u8',mode='w+',dtype=np.uint8,shape=report['source_shape'])
        active=None
        try:
            fill_ellipsoid_source(source,view);source.flush()
            # The independent direct-projector runs for this exact fixture produced
            # the fixed decoded hash below; native cleanup must preserve the solid object.
            report['expected_sha256']='3f0805dc841bf4077c60919e34858a1b9b4b860f35a8f3ba10b4f59f305ffe1f'
            t=time.perf_counter();active=RadialOwner(view,(3072,3072),OUTPUT_SHAPE)
            report['owner_setup_seconds']=time.perf_counter()-t
            report['required_device_bytes']=active.required_bytes
            for first in range(0,view.num_slices,args.chunk):
                t=time.perf_counter();device=cp.asarray(source[first:first+args.chunk]);cp.cuda.get_current_stream().synchronize()
                upload=time.perf_counter()-t
                t=time.perf_counter();active.consume(first,device)
                row={'first':first,'count':len(device),'fixture_upload_seconds':upload,'consume_seconds':time.perf_counter()-t}
                report['chunks'].append(row);del device
                print(row,flush=True)
            report['cleanup_seconds']=active.cleanup_seconds
            report['projection_seconds']=active.projection_seconds
            t=time.perf_counter();words=active.host_words();report['bitset_d2h_seconds']=time.perf_counter()-t
            active.close();active=None
            t=time.perf_counter();_d1_finalize_bitset_layer(words=words,output_shape=OUTPUT_SHAPE,
                store_dir=work/'output.cvol',model_name='fixture',view=view,projection_kind=RADIAL_OWNER_CONTRACT)
            report['publication_seconds']=time.perf_counter()-t
            report['decoded']=decode_store(work/'output.cvol')
            report['exact']=report['decoded']['sha256']==report['expected_sha256']
            (root/'owner-benchmark.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
            if not report['exact']:
                raise RuntimeError('Native owner differs from the direct projection fixture')
            print(json.dumps({k:v for k,v in report.items() if k!='chunks'}),flush=True)
        finally:
            try:
                if active is not None:active.close()
            finally:
                source._mmap.close()


if __name__=='__main__':main()
