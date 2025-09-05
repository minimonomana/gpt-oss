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

__global__ void k_rope(float *x, const float *cosv, const float *sinv,
                       int n_heads, int head_dim) {
  int h = blockIdx.x;
  int i = threadIdx.x;
  int half = head_dim / 2;
  if (i >= half || h >= n_heads) return;
  float x1 = x[h*head_dim + i];
  float x2 = x[h*head_dim + half + i];
  float c = cosv[i];
  float s = sinv[i];
  float o1 = x1 * c - x2 * s;
  float o2 = x2 * c + x1 * s;
  x[h*head_dim + i] = o1;
  x[h*head_dim + half + i] = o2;
}

static void rope_gpu(float *x, const float *cosv, const float *sinv, int n_heads, int head_dim) {
  int half = head_dim/2;
  dim3 grid(n_heads);
  dim3 block( (unsigned)half );
  hipLaunchKernelGGL(k_rope, grid, block, 0, 0, x, cosv, sinv, n_heads, head_dim);
}

static void compute_concentration_and_inv_freq_host(float base, int head_dim,
                                        float scaling_factor,
                                        float initial_context_length,
                                        float ntk_beta, float ntk_alpha,
                                        float *concentration_out,
                                        float *inv_freq_out) {
  int d_half = head_dim/2;
  float *freq = (float*)malloc(d_half*sizeof(float));
  #pragma omp parallel for
  for (int i=0;i<d_half;i++) freq[i] = powf(base, ((float)(2*i))/(float)head_dim);

  float concentration;
  if (scaling_factor > 1.0f) {
    concentration = 0.1f * logf(scaling_factor) + 1.0f;
    float low  = d_half * logf(initial_context_length / (ntk_beta * 2.0f * M_PI)) / logf(base);
    float high = d_half * logf(initial_context_length / (ntk_alpha * 2.0f * M_PI)) / logf(base);
    assert(0 < low && low < high && high < d_half-1 + 1e-3);
    #pragma omp parallel for
    for (int i=0;i<d_half;i++) {
      float interpolation = 1.0f / (scaling_factor * freq[i]);
      float extrapolation = 1.0f / freq[i];
      float ramp = ((float)i - low) / (high - low);
      if (ramp < 0) ramp = 0;
      if (ramp > 1) ramp = 1;
      float mask = 1.0f - ramp;
      inv_freq_out[i] = interpolation * (1.0f - mask) + extrapolation * mask;
    }
  } else {
    concentration = 1.0f;
    #pragma omp parallel for
    for (int i=0;i<d_half;i++) inv_freq_out[i] = 1.0f / freq[i];
  }
  *concentration_out = concentration;
  free(freq);
}

static void compute_cos_sin_host(int pos, float base, int head_dim,
                                 float scaling_factor, float initial_ctx,
                                 float ntk_beta, float ntk_alpha,
                                 float *cos_out, float *sin_out) {
  int d_half = head_dim/2;
  float conc;
  float *inv = (float*)malloc(d_half*sizeof(float));
  compute_concentration_and_inv_freq_host(base, head_dim, scaling_factor,
                                          initial_ctx, ntk_beta, ntk_alpha,
                                          &conc, inv);
  #pragma omp parallel for
  for (int j=0;j<d_half;j++) {
    float val = (float)pos * inv[j];
    cos_out[j] = cosf(val) * conc;
    sin_out[j] = sinf(val) * conc;
  }
  free(inv);
}