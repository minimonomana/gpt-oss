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


template<int TILE>
__global__ void k_gemv_rowmajor(const float * __restrict__ W,
                                const float * __restrict__ x,
                                float * __restrict__ y,
                                int in_features, int out_features) {
  // One block computes many out rows (by striding); each thread accumulates a partial
  int out = blockIdx.x * blockDim.x + threadIdx.x; // one thread per out row
  if (out >= out_features) return;

  const float *wrow = W + (size_t)out * in_features;
  float acc = 0.f;

  // tile over input dim
  for (int j0 = 0; j0 < in_features; j0 += TILE) {
    int j = j0 + threadIdx.y; // use y-threads to help coalesce x loads
    float xv = 0.f;
    if (j < in_features) xv = x[j];
    __syncthreads(); // not strictly needed with independent loads
    // each thread accumulates its own dot across the tile
    // but we simply stride by 1 here for simplicity and rely on L2
    for (int jj = j0; jj < min(j0+TILE, in_features); ++jj) {
      acc += wrow[jj] * x[jj];
    }
  }
  y[out] = acc;
}

static void gemv_gpu(float *y, const float *x, const float *W, int in_features, int out_features) {
  // grid: 1D over out features; each thread handles a row
  const int TX = 256;
  dim3 block(TX, 1, 1);
  dim3 grid((out_features + TX - 1)/TX, 1, 1);
  hipLaunchKernelGGL((k_gemv_rowmajor<1024>), grid, block, 0, 0,
                     W, x, y, in_features, out_features);
}