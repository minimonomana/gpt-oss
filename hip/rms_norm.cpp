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

#define HIP_CHECK(cmd) do { \
  hipError_t e = (cmd);     \
  if (e != hipSuccess) {    \
    fprintf(stderr, "HIP error %d (%s) at %s:%d\n", \
            (int)e, hipGetErrorString(e), __FILE__, __LINE__); \
    exit(1); \
  } \
} while(0)

__global__ void k_rms_reduce(const float *x, double *block_sums, int n) {
  extern __shared__ double ssum[];
  int tid = threadIdx.x;
  int idx = blockIdx.x * blockDim.x + tid;
  double v = 0.0;
  if (idx < n) {
    float t = x[idx];
    v = (double)t * (double)t;
  }
  ssum[tid] = v;
  __syncthreads();
  // reduce
  for (int s = blockDim.x/2; s > 0; s >>= 1) {
    if (tid < s) ssum[tid] += ssum[tid + s];
    __syncthreads();
  }
  if (tid == 0) block_sums[blockIdx.x] = ssum[0];
}

__global__ void k_scale(float *o, const float *x, const float *w, float s, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) o[i] = w[i] * (s * x[i]);
}

static void rmsnorm_gpu(float *o, const float *x, const float *weight, int n) {
  const int BS = 256;
  int nb = (n + BS - 1)/BS;
  double *d_partials;
  HIP_CHECK(hipMalloc((void**)&d_partials, nb*sizeof(double)));
  hipLaunchKernelGGL(k_rms_reduce, dim3(nb), dim3(BS), BS*sizeof(double), 0, x, d_partials, n);
  HIP_CHECK(hipDeviceSynchronize());

  // reduce on host (nb is small)
  double hsum = 0.0;
  double *hparts = (double*)malloc(nb*sizeof(double));
  HIP_CHECK(hipMemcpy(hparts, d_partials, nb*sizeof(double), hipMemcpyDeviceToHost));
  for (int i=0;i<nb;i++) hsum += hparts[i];
  free(hparts);
  HIP_CHECK(hipFree(d_partials));

  double mean = hsum / (double)n;
  double inv = 1.0 / sqrt(mean + 1e-5);
  float s = (float)inv;

  int gs = (n + BS - 1)/BS;
  hipLaunchKernelGGL(k_scale, dim3(gs), dim3(BS), 0, 0, o, x, weight, s, n);
}