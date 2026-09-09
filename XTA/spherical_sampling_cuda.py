"""FP32 virtual-cube Spherical sampling; native-T fusion is deliberately absent."""

SPHERICAL_FP32_SOURCE = r'''
__device__ __forceinline__ int clip_i(int v, int n) {
    return v < 0 ? 0 : (v >= n ? n - 1 : v);
}
__device__ __forceinline__ float clip_r(float v, float low, float high) {
    return v < low ? low : (v > high ? high : v);
}
__device__ __forceinline__ float voxel(
    const unsigned char* src, int t, int y, int x, int nt, int lt, int h, int w) {
    unsigned long long stride = (unsigned long long)h * w;
    unsigned long long at = (unsigned long long)y * w + x;
    if (nt == lt) return (float)src[(unsigned long long)t * stride + at];
    // Preserve reconstruction and gray8 rounding at each integer logical T
    // voxel before its contribution to the eight-tap shell interpolation.
    float rf = ((float)t + .5f) * ((float)nt / (float)lt) - .5f;
    int t0 = clip_i((int)floorf(rf), nt), t1 = clip_i(t0 + 1, nt);
    float alpha = clip_r(rf - (float)t0, 0.f, 1.f);
    float a = (float)src[(unsigned long long)t0 * stride + at];
    float b = (float)src[(unsigned long long)t1 * stride + at];
    return (float)__float2uint_rn(fminf(255.f, fmaxf(0.f, a + alpha * (b - a))));
}
extern "C" __global__ void spherical_virtual_fp32(
    const unsigned char* src, const float* dirs, const bool* valid,
    float* output, int nt, int lt, int h, int w,
    unsigned long long count, unsigned long long offset, float radius) {
    unsigned long long q = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (q >= count) return;
    if (!valid[q]) { output[offset + q] = 0.f; return; }
    float tt = (float)(lt - 1) * .5f + radius * dirs[3*q+2];
    float yy = (float)(h - 1) * .5f + radius * dirs[3*q+1];
    float xx = (float)(w - 1) * .5f + radius * dirs[3*q];
    if (!(tt > -1.f && tt < (float)lt && yy > -1.f && yy < (float)h &&
          xx > -1.f && xx < (float)w)) { output[offset + q] = 0.f; return; }
    int t0 = (int)floorf(tt), y0 = (int)floorf(yy), x0 = (int)floorf(xx);
    float dt = tt - (float)t0, dy = yy - (float)y0, dx = xx - (float)x0;
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
                if (ti>=0 && ti<lt && yi>=0 && yi<h && xi>=0 && xi<w)
                    value += voxel(src,ti,yi,xi,nt,lt,h,w) * ((wt*wy)*wx);
            }
        }
    }
    output[offset+q]=(float)__float2uint_rn(fminf(255.f,fmaxf(0.f,value)));
}
'''
