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

struct RopeTables {
    int max_seq;
    int head_dim;
    int half;

    float *dcos; // [max_seq, half]
    float *dsin; // [max_seq, half]
};

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

__global__ void k_compute_cos_sin_combined(int pos, float base, int head_dim,
                                           float scaling_factor, float initial_context_length,
                                           float ntk_beta, float ntk_alpha,
                                           float *cos_out, float *sin_out) {
  int d_half = head_dim / 2;
  int j = blockIdx.x * blockDim.x + threadIdx.x;
  if (j >= d_half) return;

  float freq = powf(base, ((float)(2 * j)) / (float)head_dim);

  float concentration;
  if (scaling_factor > 1.0f) {
    concentration = 0.1f * logf(scaling_factor) + 1.0f;
    float low = d_half * logf(initial_context_length / (ntk_beta * 2.0f * M_PI)) / logf(base);
    float high = d_half * logf(initial_context_length / (ntk_alpha * 2.0f * M_PI)) / logf(base);
    float interpolation = 1.0f / (scaling_factor * freq);
    float extrapolation = 1.0f / freq;
    float ramp = ((float)j - low) / (high - low);
    if (ramp < 0.0f) ramp = 0.0f;
    if (ramp > 1.0f) ramp = 1.0f;
    float mask = 1.0f - ramp;
    float inv_freq = interpolation * (1.0f - mask) + extrapolation * mask;
    float val = (float)pos * inv_freq;
    cos_out[j] = cosf(val) * concentration;
    sin_out[j] = sinf(val) * concentration;
  } else {
    concentration = 1.0f;
    float inv_freq = 1.0f / freq;
    float val = (float)pos * inv_freq;
    cos_out[j] = cosf(val) * concentration;
    sin_out[j] = sinf(val) * concentration;
  }
}

static void compute_cos_sin_gpu(int pos, float base, int head_dim,
                                         float scaling_factor, float initial_context_length,
                                         float ntk_beta, float ntk_alpha,
                                         float *cos_out, float *sin_out) {
  int d_half = head_dim / 2;
  const int TPB = 256;
  int G = (d_half + TPB - 1) / TPB;
  hipLaunchKernelGGL(k_compute_cos_sin_combined, dim3(G), dim3(TPB), 0, 0,
                     pos, base, head_dim, scaling_factor, initial_context_length,
                     ntk_beta, ntk_alpha, cos_out, sin_out);
}

__global__ void k_precompute_rope(float base, int head_dim,
                                  float scaling_factor, float initial_context_length,
                                  float ntk_beta, float ntk_alpha,
                                  int max_seq,
                                  float *cos_all, float *sin_all) {
  int half = head_dim / 2;
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= (size_t)max_seq * half) return;

  int pos = idx / half;
  int j   = idx % half;

  float freq = powf(base, ((float)(2 * j)) / (float)head_dim);
  float inv_freq, concentration;
  if (scaling_factor > 1.0f) {
    concentration = 0.1f * logf(scaling_factor) + 1.0f;
    float low = half * logf(initial_context_length / (ntk_beta * 2.0f * M_PI)) / logf(base);
    float high = half * logf(initial_context_length / (ntk_alpha * 2.0f * M_PI)) / logf(base);
    float interpolation = 1.0f / (scaling_factor * freq);
    float extrapolation = 1.0f / freq;
    float ramp = ((float)j - low) / (high - low);
    if (ramp < 0) ramp = 0;
    if (ramp > 1) ramp = 1;
    float mask = 1.0f - ramp;
    inv_freq = interpolation * (1.0f - mask) + extrapolation * mask;
  } else {
    concentration = 1.0f;
    inv_freq = 1.0f / freq;
  }

  float val = (float)pos * inv_freq;
  cos_all[pos * half + j] = cosf(val) * concentration;
  sin_all[pos * half + j] = sinf(val) * concentration;
}
