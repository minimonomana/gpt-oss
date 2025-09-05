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

__global__ void k_swiglu_gate_up(float *gate, float *up, float *out,
                                 int n, float alpha, float clampv) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  float g = gate[i];
  float u = up[i];
  // clamp
  g = fminf(g, clampv);
  u = fminf(fmaxf(u, -clampv), clampv);
  // silu
  g = g * (1.0f / (1.0f + expf(-alpha * g)));
  // up + 1
  u = u + 1.0f;
  out[i] = g * u;
}

static void swiglu_gpu(float *gate, float *up, float *out,
                       int n, float alpha, float clampv) {
  const int BS=256, GS=(n+BS-1)/BS;
  hipLaunchKernelGGL(k_swiglu_gate_up, dim3(GS), dim3(BS), 0, 0, gate, up, out, n, alpha, clampv);
}