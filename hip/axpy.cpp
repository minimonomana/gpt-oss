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

__global__ void k_axpy(float *x, const float *y, float alpha, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) x[i] += alpha * y[i];
}

static void axpy_gpu(float *x, const float *y, float alpha, int n) {
  const int BS=256, GS=(n+BS-1)/BS;
  hipLaunchKernelGGL(k_axpy, dim3(GS), dim3(BS), 0, 0, x, y, alpha, n);
}