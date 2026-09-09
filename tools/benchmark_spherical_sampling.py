"""Evaluate relaxed Spherical samplers without changing production dispatch.

The model grid, face/radius plans and frame count stay fixed. Candidate kernels
are isolated here so native-T fusion cannot silently change the shipping sampler.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import statistics
import sys

for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ.setdefault(name, '2')

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from XTA import spherical_cuda, spherical_projection
from XTA.spherical_geometry import build_spherical_view_infos, face_directions
from tools.benchmark_radial_setup import heatsoak


RELAXED_SOURCE = r'''
typedef REAL real;
__device__ __forceinline__ int clip_i(int v, int n) {
    return v < 0 ? 0 : (v >= n ? n - 1 : v);
}
__device__ __forceinline__ real clip_r(real v, real low, real high) {
    return v < low ? low : (v > high ? high : v);
}
__device__ __forceinline__ float voxel(
    const unsigned char* src, int t, int y, int x, int nt, int lt, int h, int w) {
    unsigned long long stride = (unsigned long long)h * w;
    unsigned long long at = (unsigned long long)y * w + x;
    if (FUSED_T || nt == lt) return (float)src[(unsigned long long)t * stride + at];
    real rf = ((real)t + (real).5) * ((real)nt / (real)lt) - (real).5;
    int t0 = clip_i((int)FLOOR(rf), nt), t1 = clip_i(t0 + 1, nt);
    float alpha = (float)clip_r(rf - (real)t0, (real)0, (real)1);
    float a = (float)src[(unsigned long long)t0 * stride + at];
    float b = (float)src[(unsigned long long)t1 * stride + at];
    return (float)__float2uint_rn(fminf(255.f, fmaxf(0.f, a + alpha * (b - a))));
}
extern "C" __global__ void relaxed_spherical(
    const unsigned char* src, const real* dirs, const bool* valid,
    float* output, int nt, int lt, int h, int w,
    unsigned long long count, unsigned long long offset, real radius) {
    unsigned long long q = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (q >= count) return;
    if (!valid[q]) { output[offset + q] = 0.f; return; }
    real tt = (real)(lt - 1) * (real).5 + radius * dirs[3*q+2];
    real yy = (real)(h - 1) * (real).5 + radius * dirs[3*q+1];
    real xx = (real)(w - 1) * (real).5 + radius * dirs[3*q];
    if (!(tt > (real)-1 && tt < (real)lt && yy > (real)-1 && yy < (real)h &&
          xx > (real)-1 && xx < (real)w)) { output[offset + q] = 0.f; return; }
    float t_support = 1.f;
    if (FUSED_T) {
        // Preserve the logical cube's zero border; native T itself edge-clamps.
        t_support = (float)clip_r(tt + (real)1, (real)0, (real)1);
        t_support *= (float)clip_r((real)lt - tt, (real)0, (real)1);
        tt = clip_r(tt, (real)0, (real)(lt - 1));
        tt = clip_r((tt + (real).5) * ((real)nt / (real)lt) - (real).5,
                    (real)0, (real)(nt - 1));
    }
    int t0 = (int)FLOOR(tt), y0 = (int)FLOOR(yy), x0 = (int)FLOOR(xx);
    float dt = (float)(tt - (real)t0), dy = (float)(yy - (real)y0), dx = (float)(xx - (real)x0);
    float value = 0.f;
    #pragma unroll
    for (int it=0; it<2; ++it) {
        int ti=t0+it; float wt=it ? dt : 1.f-dt;
        #pragma unroll
        for (int iy=0; iy<2; ++iy) {
            int yi=y0+iy; float wy=iy ? dy : 1.f-dy;
            #pragma unroll
            for (int ix=0; ix<2; ++ix) {
                int xi=x0+ix; float wx=ix ? dx : 1.f-dx;
                if (ti>=0 && ti<(FUSED_T ? nt : lt) && yi>=0 && yi<h && xi>=0 && xi<w)
                    value += voxel(src,ti,yi,xi,nt,lt,h,w) * ((wt*wy)*wx);
            }
        }
    }
    output[offset+q]=(float)__float2uint_rn(fminf(255.f,fmaxf(0.f,value*t_support)));
}
'''


def kernels(cp):
    reference = cp.RawModule(code=spherical_cuda._KERNEL_SOURCE,
                             options=('--std=c++11', '--fmad=false'))
    result = {'reference': (reference.get_function('spherical_native_f32'), np.float64)}
    for name, real, fused in (('fma64', 'double', 0), ('fp32', 'float', 0), ('native_t_fp32', 'float', 1)):
        code = RELAXED_SOURCE.replace('REAL', real).replace('FUSED_T', str(fused)).replace(
            'FLOOR', 'floor' if real == 'double' else 'floorf')
        module = cp.RawModule(code=code, options=('--std=c++11', '--fmad=true'))
        result[name] = (module.get_function('relaxed_spherical'), np.float64 if real == 'double' else np.float32)
    return result


def render(cp, compiled, name, source, view, radius, directions):
    kernel, dtype = compiled[name]
    rays, valid = directions[dtype]
    output = cp.empty(valid.shape, cp.float32)
    kernel(((valid.size + 255)//256,), (256,), (source, rays, valid, output,
        np.int32(source.shape[0]), np.int32(view.full_t), np.int32(view.full_h), np.int32(view.full_w),
        np.uint64(valid.size), np.uint64(0), dtype(radius)))
    return output


def directions_for(cp, view):
    key = ('cuda:0', view.spherical_face, view.spherical_face_intervals,
           view.src_h, view.src_w, view.spherical_u_origin, view.spherical_v_origin,
           tuple(view.spherical_rotation_xyz))
    rays = np.empty((view.src_h, view.src_w, 3), np.float64)
    valid = np.empty((view.src_h, view.src_w), np.bool_)
    for row in range(0, view.src_h, 64):
        stop = min(row + 64, view.src_h)
        rays[row:stop], valid[row:stop] = spherical_cuda._build_host_direction_block(key, row, stop)
    return {dtype: (cp.asarray(rays, dtype=dtype), cp.asarray(valid)) for dtype in (np.float64, np.float32)}


def errors(reference, candidate, valid):
    delta = np.abs(candidate[valid] - reference[valid])
    return {'pixels': int(delta.size), 'changed': int(np.count_nonzero(delta)),
            'max_abs': float(delta.max(initial=0)), 'mae': float(delta.mean()) if delta.size else 0.,
            'rmse': float(np.sqrt(np.mean(delta**2))) if delta.size else 0.,
            'p99_abs': float(np.percentile(delta, 99)) if delta.size else 0.}


def quality(cp, compiled):
    shape = (25, 65, 67)
    views = build_spherical_view_infos(*shape, targets=('transverse',), min_radius=1., patch_size=64, tilted_views=())
    cases = {}
    source_shape = (17, shape[1], shape[2])
    random = np.random.default_rng(912).integers(0, 256, source_shape, np.uint8)
    sheet = np.zeros(source_shape, np.uint8); sheet[8, 23:43, 24:44] = 255
    rods = np.zeros(source_shape, np.uint8); rods[8, 27, 21:47] = 255; rods[9, 34:36, 21:47] = 255
    cases.update(random=random, one_voxel_sheet=sheet, thin_rods=rods)
    results = []
    for label, host in cases.items():
        source = cp.asarray(host)
        for view in views:
            directions = directions_for(cp, view)
            valid = directions[np.float64][1].get()
            for radius in (view.spherical_radii[2], view.spherical_radii[-1]):
                reference = render(cp, compiled, 'reference', source, view, radius, directions).get()
                for name in ('fma64', 'fp32', 'native_t_fp32'):
                    candidate = render(cp, compiled, name, source, view, radius, directions).get()
                    results.append({'case': label, 'face': view.spherical_face, 'radius': radius,
                                    'variant': name, **errors(reference, candidate, valid)})
    return results


def benchmark(cp, compiled, repeats):
    views = build_spherical_view_infos(2911, 3064, 3022, targets=('transverse',),
                                      min_radius=244.5, patch_size=3072, tilted_views=())
    view = views[0]
    source = cp.random.RandomState(913).randint(0, 256, (128, 3064, 3022), dtype=cp.uint8)
    directions = directions_for(cp, view)
    valid = directions[np.float64][1].get()
    results = []
    for radius in (400., view.spherical_max_radius):
        reference = render(cp, compiled, 'reference', source, view, radius, directions).get()
        timings = {name: [] for name in compiled}
        quality_rows = {}
        for name in compiled:
            candidate = render(cp, compiled, name, source, view, radius, directions).get()
            quality_rows[name] = errors(reference, candidate, valid)
        for iteration in range(repeats):
            order = list(compiled) if iteration % 2 == 0 else list(compiled)[::-1]
            for name in order:
                # Reuse the output while timing kernel work, excluding allocation.
                output = cp.empty(valid.shape, cp.float32)
                kernel, dtype = compiled[name]; rays, mask = directions[dtype]
                begin, end = cp.cuda.Event(), cp.cuda.Event()
                begin.record()
                for _ in range(5):
                    kernel(((mask.size+255)//256,), (256,), (source,rays,mask,output,
                        np.int32(source.shape[0]),np.int32(view.full_t),np.int32(view.full_h),np.int32(view.full_w),
                        np.uint64(mask.size),np.uint64(0),dtype(radius)))
                end.record(); end.synchronize()
                timings[name].append(cp.cuda.get_elapsed_time(begin,end)/5)
        for name in compiled:
            results.append({'radius':radius,'variant':name,'median_kernel_ms':statistics.median(timings[name]),
                            'samples_ms':timings[name],**quality_rows[name]})
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--heat-seconds', type=float, default=60.)
    parser.add_argument('--repeats', type=int, default=7)
    args = parser.parse_args()
    args.output = args.output.resolve()
    if args.output.is_relative_to(ROOT) or args.repeats < 1 or args.heat_seconds < 0:
        parser.error('Use task Scratch output, positive repeats and nonnegative heat seconds')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault('CUPY_CACHE_DIR', str(args.output.parent/'cupy-cache'))
    import cupy as cp
    compiled = kernels(cp)
    report = {'scope': __doc__, 'device': cp.cuda.runtime.getDeviceProperties(0)['name'].decode(),
              'quality': quality(cp, compiled)}
    report['heatsoak_seconds'] = heatsoak(args.heat_seconds, 0)
    report['benchmark'] = benchmark(cp, compiled, args.repeats)
    report['limits'] = ('Kernel timing uses production working dimensions but only128 native T slices to fit the local GPU. '
                       '17-to25 quality fixtures preserve a production-like T ratio. No model inference or accuracy claim.')
    args.output.write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
    for row in report['benchmark']:
        print(json.dumps({k:v for k,v in row.items() if k!='samples_ms'}), flush=True)


if __name__ == '__main__':
    main()
