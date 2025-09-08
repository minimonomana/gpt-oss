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

// warp reduce max
__inline__ __device__ float warp_reduce_max(float v) {
    for (int offset = 16; offset > 0; offset >>= 1)
        v = fmaxf(v, __shfl_down(v, offset));
    return v;
}

// warp reduce sum
__inline__ __device__ float warp_reduce_sum_softmax(float v) {
    for (int offset = 16; offset > 0; offset >>= 1)
        v += __shfl_down(v, offset);
    return v;
}

__global__ void k_softmax_rows(float *att, int row_len, int row_stride) {
    int head = blockIdx.x;
    float *row = att + (size_t)head * row_stride;

    int tid = threadIdx.x;
    int lane = tid & 31;
    int warp = tid >> 5;

    // Step 1: find max
    float local_max = -INFINITY;
    for (int i = tid; i < row_len; i += blockDim.x) {
        local_max = fmaxf(local_max, row[i]);
    }
    float warp_max = warp_reduce_max(local_max);

    __shared__ float block_max[8]; // supports up to 8 warps (256 threads)
    if (lane == 0) block_max[warp] = warp_max;
    __syncthreads();
    float mx = -INFINITY;
    if (warp == 0) {
        float val = (tid < (blockDim.x>>5)) ? block_max[lane] : -INFINITY;
        float wmax = warp_reduce_max(val);
        if (lane == 0) block_max[0] = wmax;
    }
    __syncthreads();
    mx = block_max[0];

    // Step 2: exp and local sum
    float local_sum = 0.f;
    for (int i = tid; i < row_len; i += blockDim.x) {
        float v = __expf(row[i] - mx);
        row[i] = v;
        local_sum += v;
    }
    float warp_sum = warp_reduce_sum_softmax(local_sum);
    if (lane == 0) block_max[warp] = warp_sum;
    __syncthreads();
    float sum = 0.f;
    if (warp == 0) {
        float val = (tid < (blockDim.x>>5)) ? block_max[lane] : 0.f;
        float wsum = warp_reduce_sum_softmax(val);
        if (lane == 0) block_max[0] = wsum;
    }
    __syncthreads();
    sum = block_max[0];

    // Step 3: normalize
    for (int i = tid; i < row_len; i += blockDim.x) {
        row[i] = row[i] / sum;
    }
}

static void softmax_rows_gpu(float *att, int n_heads, int row_len, int row_stride) {
    int BS = 256; // good for row_len up to 2048
    dim3 grid(n_heads);
    hipLaunchKernelGGL(k_softmax_rows, grid, dim3(BS), 0, 0, att, row_len, row_stride);
}
