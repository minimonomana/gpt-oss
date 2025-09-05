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

__global__ void k_add_bias(float *y, const float *b, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) y[i] += b[i];
}

static void add_bias_gpu(float *y, const float *b, int n) {
  const int BS=256, GS=(n+BS-1)/BS;
  hipLaunchKernelGGL(k_add_bias, dim3(GS), dim3(BS), 0, 0, y, b, n);
}