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

__global__ void k_softmax_row(float *att, int row_len) {
  // one block per row, 1D thread over len
  extern __shared__ float sh[];
  float *buf = sh; // for reductions
  int tid = threadIdx.x;
  int n = row_len;

  // find max
  float mx = -INFINITY;
  for (int i = tid; i < n; i += blockDim.x) {
    mx = fmaxf(mx, att[i]);
  }
  buf[tid] = mx;
  __syncthreads();
  for (int s = blockDim.x/2; s>0; s>>=1) {
    if (tid < s) buf[tid] = fmaxf(buf[tid], buf[tid+s]);
    __syncthreads();
  }
  mx = buf[0];

  // exp & sum
  float sum = 0.f;
  for (int i = tid; i < n; i += blockDim.x) {
    float v = expf(att[i] - mx);
    att[i] = v;
    sum += v;
  }
  buf[tid] = sum;
  __syncthreads();
  for (int s = blockDim.x/2; s>0; s>>=1) {
    if (tid < s) buf[tid] += buf[tid+s];
    __syncthreads();
  }
  sum = buf[0];

  // normalize
  for (int i = tid; i < n; i += blockDim.x) {
    att[i] = att[i] / sum;
  }
}

static void softmax_rows_gpu(float *att, int n_heads, int row_len, int row_stride) {
  const int BS = 256;
  for (int h=0; h<n_heads; ++h) {
    float *row = att + (size_t)h * row_stride; 
    hipLaunchKernelGGL(k_softmax_row, dim3(1), dim3(BS), BS*sizeof(float), 0, row, row_len);
  }
}