"""Run the real TTA worker/scheduler for all shell families using a local PT model.

The sole validation adaptation removes the older physical families after view
compilation: their D1 path requires TensorRT, unavailable in the local PT runtime.
No shell geometry, inference, cleanup, output or scheduler operation is mocked.
"""
import argparse
from dataclasses import replace
import os
from pathlib import Path
import sys
from unittest import mock

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,required=True)
    parser.add_argument('--model',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--owner',choices=('0','1'),required=True)
    parser.add_argument('--imgsz',type=int,default=256)
    args=parser.parse_args()
    os.environ['YOLO_TTA_RADIAL_OWNER']=args.owner
    os.environ['YOLO_TTA_GPU_WORKER_MIN_LEASE_SLICES']='8'
    os.environ['YOLO_TTA_GPU_WORKER_MAX_LEASE_SLICES']='16'
    os.environ['YOLO_TTA_TELEMETRY']='0'
    from XTA import pipeline
    original=pipeline.compile_physical_views
    def shell_views(**kwargs):
        compiled=original(**kwargs)
        views=tuple(v for v in compiled.views if v.family=='radial')
        print(f'Qualification scope: {len(views)} shell trajectories; older physical families excluded for PT-only runtime.',flush=True)
        return replace(compiled,views=views)
    sys.argv=['xta','--input',str(args.input),'--model','gpu:'+str(args.model),
        '--conf','0.5','--min_conf','0','--angle','0','--imgsz',str(args.imgsz),'--min_radius','0',
        '--postprocessing','keep_objects:1','--enable_tilted','transverse','sagittal','coronal',
        '--enable_radial','transverse','sagittal','coronal','tilted_transverse','tilted_sagittal','tilted_coronal',
        '--output',str(args.output_dir),'--temp',str(args.output_dir.parent/(args.output_dir.name+'-temp')),
        '--save','low_quality:0.20','nrrd','summary','--batch','gpu:1','--device','0',
        '--quantize','gpu:fp16','--channel_format','grey','--interpolation_distance','0']
    with mock.patch.object(pipeline,'compile_physical_views',shell_views):
        pipeline.main()


if __name__=='__main__':main()
