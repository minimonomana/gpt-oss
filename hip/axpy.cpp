#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <omp.h>

#include <assert.h>
#include <ctype.h>
#include <fcntl.h>
#include <limits.h>
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// __global__ void k_axpy(float *x, const float *y, float alpha, int n) {
//   int i = blockIdx.x * blockDim.x + threadIdx.x;
//   if (i < n) x[i] += alpha * y[i];
// }

// static void axpy_gpu(float *x, const float *y, float alpha, int n) {
//   const int BS=256, GS=(n+BS-1)/BS;
//   hipLaunchKernelGGL(k_axpy, dim3(GS), dim3(BS), 0, 0, x, y, alpha, n);
// }

__global__ void k_axpy_vec(float * __restrict__ x,
                           const float * __restrict__ y,
                           float alpha, int n) {
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 4;

    if (idx + 3 < n) {
        float4 vx = reinterpret_cast<float4*>(x)[idx/4];
        float4 vy = reinterpret_cast<const float4*>(y)[idx/4];
        vx.x += alpha * vy.x;
        vx.y += alpha * vy.y;
        vx.z += alpha * vy.z;
        vx.w += alpha * vy.w;
        reinterpret_cast<float4*>(x)[idx/4] = vx;
    } else {
        for (int i = idx; i < n; i++) {
            x[i] += alpha * y[i];
        }
    }
}

static void axpy_gpu(float *x, const float *y, float alpha, int n) {
    const int THREADS = 256;
    int nblocks = (n + THREADS - 1) / (THREADS);
    hipLaunchKernelGGL(k_axpy_vec, dim3(nblocks), dim3(THREADS), 0, 0, x, y, alpha, n);
}