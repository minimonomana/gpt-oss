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

// __global__ void k_add_bias(float *y, const float *b, int n) {
//   int i = blockIdx.x * blockDim.x + threadIdx.x;
//   if (i < n) y[i] += b[i];
// }

// static void add_bias_gpu(float *y, const float *b, int n) {
//   const int BS=256, GS=(n+BS-1)/BS;
//   hipLaunchKernelGGL(k_add_bias, dim3(GS), dim3(BS), 0, 0, y, b, n);
// }

__global__ void k_add_bias_vec(float * __restrict__ y,
                               const float * __restrict__ b,
                               int n) {
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 4; // process 4 elements per thread

    if (idx + 3 < n) {
        float4 vy = reinterpret_cast<float4*>(y)[idx/4];
        float4 vb = reinterpret_cast<const float4*>(b)[idx/4];
        vy.x += vb.x;
        vy.y += vb.y;
        vy.z += vb.z;
        vy.w += vb.w;
        reinterpret_cast<float4*>(y)[idx/4] = vy;
    } else {
        for (int i = idx; i < n; i++) {
            y[i] += b[i];
        }
    }
}

static void add_bias_gpu(float *y, const float *b, int n) {
    const int THREADS = 256;
    int nblocks = (n + THREADS - 1) / (THREADS);
    hipLaunchKernelGGL(k_add_bias_vec, dim3(nblocks), dim3(THREADS), 0, 0, y, b, n);
}
