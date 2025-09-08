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

// GPU kernel for concentration and inv_freq
__global__ void k_compute_concentration_and_inv_freq(
    float base, int head_dim, float scaling_factor, float initial_context_length,
    float ntk_beta, float ntk_alpha, float *concentration_out, float *inv_freq_out) {
  int d_half = head_dim / 2;
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i >= d_half) return;

  float freq = powf(base, ((float)(2 * i)) / (float)head_dim);
  float concentration;
  if (scaling_factor > 1.0f) {
    concentration = 0.1f * logf(scaling_factor) + 1.0f;
    float low = d_half * logf(initial_context_length / (ntk_beta * 2.0f * M_PI)) / logf(base);
    float high = d_half * logf(initial_context_length / (ntk_alpha * 2.0f * M_PI)) / logf(base);
    float interpolation = 1.0f / (scaling_factor * freq);
    float extrapolation = 1.0f / freq;
    float ramp = ((float)i - low) / (high - low);
    if (ramp < 0) ramp = 0;
    if (ramp > 1) ramp = 1;
    float mask = 1.0f - ramp;
    inv_freq_out[i] = interpolation * (1.0f - mask) + extrapolation * mask;
    if (i == 0) *concentration_out = concentration;
  } else {
    concentration = 1.0f;
    inv_freq_out[i] = 1.0f / freq;
    if (i == 0) *concentration_out = concentration;
  }
}

// GPU kernel for cos/sin computation
__global__ void k_compute_cos_sin(int pos, float concentration, const float *inv_freq, float *cos_out, float *sin_out, int d_half) {
  int j = threadIdx.x + blockIdx.x * blockDim.x;
  if (j >= d_half) return;
  float val = (float)pos * inv_freq[j];
  cos_out[j] = cosf(val) * concentration;
  sin_out[j] = sinf(val) * concentration;
}

// GPU wrapper for compute_cos_sin
void compute_cos_sin_gpu(int pos, float base, int head_dim, float scaling_factor,
                         float initial_context_length, float ntk_beta, float ntk_alpha,
                         float *cos_out, float *sin_out) {
  int d_half = head_dim / 2;
  float *d_inv_freq, *d_concentration;
  HIP_CHECK(hipMalloc(&d_inv_freq, d_half * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_concentration, sizeof(float)));

  // Launch kernel to compute concentration and inv_freq
  k_compute_concentration_and_inv_freq<<<(d_half+255)/256, 256>>>(
      base, head_dim, scaling_factor, initial_context_length,
      ntk_beta, ntk_alpha, d_concentration, d_inv_freq);
  HIP_CHECK(hipDeviceSynchronize());

  // Copy concentration to host
  float concentration;
  HIP_CHECK(hipMemcpy(&concentration, d_concentration, sizeof(float), hipMemcpyDeviceToHost));

  // Launch kernel to compute cos/sin
  k_compute_cos_sin<<<(d_half+255)/256, 256>>>(
      pos, concentration, d_inv_freq, cos_out, sin_out, d_half);
  HIP_CHECK(hipDeviceSynchronize());

  HIP_CHECK(hipFree(d_inv_freq));
  HIP_CHECK(hipFree(d_concentration));
}