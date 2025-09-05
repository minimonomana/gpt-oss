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

__global__ void k_attn_scores(const float *q, const float *k_cache,
                              const float *mask, float *att,
                              int head_dim, int kv_mul,
                              int seq_len, int pos, int kv_dim,
                              int n_heads, int sliding_window, int apply_sw_mask) {
  int h = blockIdx.x;     // head
  int t = threadIdx.x;    // timestep 0..pos (+1 for sink later)
  if (h >= n_heads || t > pos) return;

  const float *qh = q + h*head_dim;
  const float *kt = k_cache + t*kv_dim + (h/kv_mul)*head_dim;

  // dot
  float s = 0.f;
  #pragma unroll
  for (int i=0;i<head_dim;i++) s += qh[i]*kt[i];
  s /= sqrtf((float)head_dim);

  if (apply_sw_mask) {
    // mask[pos, t] already -inf where outside window
    s += mask[pos * seq_len + t];
  }
  att[h*(seq_len+1) + t] = s;
}

__global__ void k_attn_weighted_sum(const float *att, const float *v_cache,
                                    float *tb, int head_dim, int kv_mul,
                                    int row_stride, int row_len, int kv_dim,
                                    int n_heads) {
  int h = blockIdx.x;          // head
  int i = threadIdx.x;         // dim element
  if (h >= n_heads || i >= head_dim) return;

  float out = 0.f;
  // row_len == pos+2 (tokens 0..pos plus sink)
  for (int t = 0; t < row_len; ++t) {
    const float a = att[(size_t)h * row_stride + t];
    const float *vt = v_cache + (size_t)t * kv_dim + (size_t)(h / kv_mul) * head_dim;
    out += a * vt[i];
  }
  tb[(size_t)h * head_dim + i] = out;
}

static void attn_scores_gpu(const float *q, const float *k_cache,
                            const float *mask, float *att,
                            int head_dim, int kv_mul, int seq_len, int pos,
                            int kv_dim, int n_heads, int sliding_window) {
  int apply = (sliding_window > 0) ? 1 : 0;
  // one block per head; threads over time steps (pad to warp multiple)
  int nthreads = 1; while (nthreads < (pos+1)) nthreads <<=1;
  if (nthreads < 64) nthreads = 64; // wavefront
  hipLaunchKernelGGL(k_attn_scores, dim3(n_heads), dim3(nthreads), 0, 0,
                     q, k_cache, mask, att,
                     head_dim, kv_mul, seq_len, pos, kv_dim, n_heads,
                     sliding_window, apply);
}

static void attn_weighted_sum_gpu(const float *att, const float *v_cache,
                                  float *tb, int head_dim, int kv_mul,
                                  int seq_len, int pos, int kv_dim, int n_heads) {
  int row_len = pos + 2;           // include sink
  int row_stride = seq_len + 1;    // physical stride used when storing att
  dim3 grid(n_heads);
  dim3 block((unsigned)head_dim);
  hipLaunchKernelGGL(k_attn_weighted_sum, grid, block, 0, 0,
                     att, v_cache, tb, head_dim, kv_mul,
                     row_stride, row_len, kv_dim, n_heads);
}