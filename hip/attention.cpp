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

__global__ void k_attn_scores_shared(const float *q, const float *k_cache,
                                     const float *mask, float *att,
                                     int head_dim, int kv_mul,
                                     int seq_len, int pos, int kv_dim,
                                     int n_heads, int apply_sw_mask) {
  extern __shared__ float q_sh[]; // size = head_dim
  int h = blockIdx.x;
  if (h >= n_heads) return;

  const float *qh = q + (size_t)h * head_dim;
  // load q into shared memory (thread-strided)
  for (int i = threadIdx.x; i < head_dim; i += blockDim.x) {
    q_sh[i] = qh[i];
  }
  __syncthreads();

  // each thread computes scores for t = threadIdx.x, threadIdx.x + blockDim.x, ...
  for (int t = threadIdx.x; t <= pos; t += blockDim.x) {
    const float *kt = k_cache + (size_t)t * kv_dim + (size_t)(h / kv_mul) * head_dim;
    float s = 0.f;
    // accumulate (head_dim small-ish, loop in registers using q_sh)
    for (int i = 0; i < head_dim; ++i) {
      s += q_sh[i] * kt[i];
    }
    s /= sqrtf((float)head_dim);
    if (apply_sw_mask && mask) {
      s += mask[pos * seq_len + t]; // mask should be -inf for out-of-window
    }
    att[(size_t)h * (seq_len + 1) + t] = s;
  }
}

static void attn_scores_gpu(const float *q, const float *k_cache,
                                const float *mask, float *att,
                                int head_dim, int kv_mul, int seq_len, int pos,
                                int kv_dim, int n_heads, int apply_sw_mask) {
  int threads = min(256, head_dim);                 // good default
  int g = n_heads;
  size_t shared_mem = head_dim * sizeof(float);
  hipLaunchKernelGGL(k_attn_scores_shared, dim3(g), dim3(threads), shared_mem, 0,
                     q, k_cache, mask, att, head_dim, kv_mul, seq_len, pos, kv_dim, n_heads, apply_sw_mask);
}

__global__ void k_attn_weighted_sum_strided(const float *att, const float *v_cache,
                                            float *tb, int head_dim, int kv_mul,
                                            int row_stride, int row_len, int kv_dim,
                                            int n_heads) {
  extern __shared__ float att_sh[]; // row_len floats per block (but row_len <= seq_len+1)
  int h = blockIdx.x;
  if (h >= n_heads) return;

  // load attention row into shared mem (single block)
  for (int j = threadIdx.x; j < row_len; j += blockDim.x)
    att_sh[j] = att[(size_t)h * row_stride + j];
  __syncthreads();

  int i = threadIdx.x;
  // each thread processes multiple output dims
  for (int dim_idx = i; dim_idx < head_dim; dim_idx += blockDim.x) {
    float out = 0.f;
    // iterate t
    for (int t = 0; t < row_len; ++t) {
      float a = att_sh[t];
      const float *vt = v_cache + (size_t)t * kv_dim + (size_t)(h / kv_mul) * head_dim;
      out += a * vt[dim_idx];
    }
    tb[(size_t)h * head_dim + dim_idx] = out;
  }
}

static void attn_weighted_sum_gpu(const float *att, const float *v_cache,
                                      float *tb, int head_dim, int kv_mul,
                                      int seq_len, int pos, int kv_dim, int n_heads) {
  int row_len = pos + 1;
  int row_stride = seq_len + 1;
  int threads = min(256, head_dim);
  size_t shared = row_len * sizeof(float);
  hipLaunchKernelGGL(k_attn_weighted_sum_strided, dim3(n_heads), dim3(threads), shared, 0,
                     att, v_cache, tb, head_dim, kv_mul, row_stride, row_len, kv_dim, n_heads);
}
