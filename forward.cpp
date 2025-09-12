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

#include "hip/rms_norm.cpp"
#include "hip/matmul.cpp"
#include "hip/softmax.cpp"
#include "hip/add_bias.cpp"
#include "hip/rope.cpp"
#include "hip/attention.cpp"
#include "hip/swiglu.cpp"
#include "hip/axpy.cpp"

extern RopeTables g_rope_tables;
extern __half *d_w_mlp1_fp16;

#define HIP_CHECK(cmd) do { \
  hipError_t e = (cmd);     \
  if (e != hipSuccess) {    \
    fprintf(stderr, "HIP error %d (%s) at %s:%d\n", \
            (int)e, hipGetErrorString(e), __FILE__, __LINE__); \
    exit(1); \
  } \
} while(0)

__global__ void k_set(float *x, float v, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) x[i] = v;
}

__global__ void k_split_qkv(const float *qkv, float *q, float *k, float *v,
                            int head_dim, int n_q, int n_kv) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int total = head_dim * (n_q + 2*n_kv);
  if (i >= total) return;
  if (i < head_dim * n_q) {
    q[i] = qkv[i];
  } else if (i < head_dim * (n_q + n_kv)) {
    k[i - head_dim * n_q] = qkv[i];
  } else {
    v[i - head_dim * (n_q + n_kv)] = qkv[i];
  }
}

static void split_qkv_gpu(const float *qkv, float *q, float *k, float *v,
                          int head_dim, int n_q, int n_kv) {
  int total = head_dim*(n_q + 2*n_kv);
  const int BS=256, GS=(total+BS-1)/BS;
  hipLaunchKernelGGL(k_split_qkv, dim3(GS), dim3(BS), 0, 0, qkv, q, k, v,
                     head_dim, n_q, n_kv);
}

__global__ void split_gate_up(const float *mlp1_out, float *gate, float *up, int intermediate_dim) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < intermediate_dim) {
        gate[i] = mlp1_out[i*2];
        up[i]   = mlp1_out[i*2 + 1];
    }
}

__global__ void k_append_sink(float *att, const float *sinks, int seq_len, int pos_plus1, int n_heads) {
    int h = blockIdx.x * blockDim.x + threadIdx.x;
    if (h < n_heads) {
        att[h * (seq_len + 1) + pos_plus1] = sinks[h];
    }
}

// Kernel to initialize Pair array from values
__global__ void k_init_pairs(Pair *pairs, const float *values, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        pairs[i].value = values[i];
        pairs[i].index = i;
    }
}

// Kernel to find top-k values and indices (simple selection, for small k)
__global__ void k_topk(Pair *pairs, float *topk_values, int *topk_indices, int n, int k) {
    // Only one thread does the selection (for small k)
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        // Simple partial selection sort
        for (int i = 0; i < k; ++i) {
            int max_idx = i;
            for (int j = i + 1; j < n; ++j) {
                if (pairs[j].value > pairs[max_idx].value) {
                    max_idx = j;
                }
            }
            // Swap
            if (max_idx != i) {
                Pair tmp = pairs[i];
                pairs[i] = pairs[max_idx];
                pairs[max_idx] = tmp;
            }
            topk_values[i] = pairs[i].value;
            topk_indices[i] = pairs[i].index;
        }
    }
}

// GPU topk function
void topk_gpu(float *topk_values, int *topk_indices, const float *router_score, int num_experts, int experts_per_token) {
    // Allocate device memory for pairs
    Pair *d_pairs;
    HIP_CHECK(hipMalloc(&d_pairs, num_experts * sizeof(Pair)));

    // Initialize pairs on device
    int BS = 256, GS = (num_experts + BS - 1) / BS;
    hipLaunchKernelGGL(k_init_pairs, dim3(GS), dim3(BS), 0, 0, d_pairs, router_score, num_experts);

    // Find top-k on device
    hipLaunchKernelGGL(k_topk, dim3(1), dim3(1), 0, 0, d_pairs, topk_values, topk_indices, num_experts, experts_per_token);

    HIP_CHECK(hipFree(d_pairs));
}

static float* forward_gpu(Transformer *T, int token, int pos) {
  const Config &p = T->config;
  const TransformerWeights &w = T->weights;
  RunState &s = T->state;

  const int H = p.hidden_dim;
  const int D = p.head_dim;
  const int Hq = p.n_attn_heads;
  const int Hkv = p.n_kv_heads;
  const int kv_dim = D * Hkv;
  const int kv_mul = Hq / Hkv;

  // x = embedding[token]
  // We'll just gemv against an implicit one-hot by copying that row:
  // Copy happens on device: y = token_embedding_table[token, :]
  const float *emb_row = w.token_embedding_table + (size_t)token*H;
  HIP_CHECK(hipMemcpy(s.x, emb_row, H*sizeof(float), hipMemcpyDeviceToDevice));

  for (int l=0; l<p.n_layers; ++l) {
    // --- Attention RMSNorm: t = rmsnorm(x, rms_attn_w[l])
    rmsnorm_gpu(s.t, s.x, w.rms_attn_w + (size_t)l*H, H);

    // --- QKV projection: qkv = W_qkv[l] * t + b_qkv[l]
    const float *Wqkv = w.w_qkv + (size_t)l*H*(D*Hq + 2*D*Hkv);
    const float *Bqkv = w.b_qkv + (size_t)l*(D*Hq + 2*D*Hkv);
    gemv_gpu(s.qkv, s.t, Wqkv, H, (D*Hq + 2*D*Hkv));
    add_bias_gpu(s.qkv, Bqkv, (D*Hq + 2*D*Hkv));

    // split to q,k,v. k/v current position also appended into cache
    s.k = T->state.key_cache + (size_t)l*p.seq_len*kv_dim + (size_t)pos*kv_dim;
    s.v = T->state.value_cache + (size_t)l*p.seq_len*kv_dim + (size_t)pos*kv_dim;
    split_qkv_gpu(s.qkv, s.q, s.k, s.v, D, Hq, Hkv);

    // --- RoPE for q and k(pos)
    int half = D/2;
    const float *dcos = g_rope_tables.dcos + pos * half;
    const float *dsin = g_rope_tables.dsin + pos * half;
    rope_gpu(s.q, dcos, dsin, Hq, D);
    rope_gpu(s.k, dcos, dsin, Hkv, D);
    // HIP_CHECK(hipFree(dcos)); HIP_CHECK(hipFree(dsin));

    // --- Attention scores for all heads vs 0..pos
    float *k_layer_cache = T->state.key_cache + (size_t)l*p.seq_len*kv_dim;
    float *v_layer_cache = T->state.value_cache + (size_t)l*p.seq_len*kv_dim;
    attn_scores_gpu(s.q, k_layer_cache, (p.sliding_window>0 && (l % 2 == 0))? s.mask: nullptr, s.att,
                    D, kv_mul, p.seq_len, pos, kv_dim, Hq, (l%2==0)?p.sliding_window:0);

    // write sink score at index pos+1 (per head) from attn_sinks
    {
      const int BS = 256, GS = (Hq + BS - 1) / BS;
      const float *sink_ptr = w.attn_sinks + (size_t)l * Hq;
      hipLaunchKernelGGL(k_append_sink, dim3(GS), dim3(BS), 0, 0,
                        s.att, sink_ptr, p.seq_len, pos + 1, Hq);
      // HIP_CHECK(hipDeviceSynchronize());
    }

    // softmax over len = pos+2 for each head
    softmax_rows_gpu(s.att, Hq, pos + 2, p.seq_len + 1);

    // weighted sum over V → tb(heads*D)
    attn_weighted_sum_gpu(s.att, v_layer_cache, s.tb, D, kv_mul, p.seq_len, pos, kv_dim, Hq);

    // output projection: tb2 = W_o[l] * tb + b_o[l]
    const float *Wo = w.w_o + (size_t)l*(D*Hq)*H;
    const float *Bo = w.b_o + (size_t)l*H;
    gemv_gpu(s.tb2, s.tb, Wo, D*Hq, H);
    add_bias_gpu(s.tb2, Bo, H);

    // residual: x += tb2
    axpy_gpu(s.x, s.tb2, 1.0f, H);

    // --- MLP: t = rmsnorm(x, rms_ffn_w[l])
    rmsnorm_gpu(s.t, s.x, w.rms_ffn_w + (size_t)l*H, H);

    // router: router = W_router * t + b_router
    const float *Wr = w.w_router + (size_t)l*H*p.n_experts;
    const float *Br = w.b_router + (size_t)l*p.n_experts;
    gemv_gpu(s.router_score, s.t, Wr, H, p.n_experts);
    add_bias_gpu(s.router_score, Br, p.n_experts);
    // HIP_CHECK(hipDeviceSynchronize()); // ensure router on device finished

    topk_gpu(s.topk_v, s.topk_i, s.router_score, p.n_experts, p.experts_per_token);
    softmax_rows_gpu(s.topk_v, 1, p.experts_per_token, p.experts_per_token); // normalize top-k values

    // e_agg = 0
    {
      const int BS=256, GS=(H+BS-1)/BS;
      hipLaunchKernelGGL(k_set, dim3(GS), dim3(BS), 0, 0, s.e_agg, 0.f, H);
    }

    // For each selected expert e:
    for (int idx=0; idx<p.experts_per_token; ++idx) {
      int e = s.topk_i[idx];
      float wexp = s.topk_v[idx];
      if (e < 0) continue;

      // MLP1: mlp1_out = W_mlp1[l,e] * t + b_mlp1[l,e] → size 2*intermediate
      // const float *W1 = w.w_mlp1 + (size_t)(l*p.n_experts + e) * (2*p.intermediate_dim) * H;
      // const void *W_mlp1_fp16 = (w.w_mlp1) + (size_t)(l*p.n_experts + e) * (2*p.intermediate_dim) * H;
      // const float *B1 = w.b_mlp1 + (size_t)(l*p.n_experts + e) * (2*p.intermediate_dim);
      // // gemv_gpu(s.mlp1_out, s.t, W1, H, 2*p.intermediate_dim);
      // gemv_fp16_weights(s.mlp1_out, s.t, W_mlp1_fp16, H, 2 * p.intermediate_dim);
      // add_bias_gpu(s.mlp1_out, B1, 2*p.intermediate_dim);

      // const __half *w_mlp1_half = (const __half *)( (const void*) d_w_mlp1_fp16 );
      size_t expert_index = (size_t)l * (size_t)p.n_experts + (size_t)e;
      __half *W1_fp16 = d_w_mlp1_fp16 + expert_index * ( (size_t)2 * (size_t)p.intermediate_dim * (size_t)H );

      gemv_fp16_weights(s.mlp1_out, s.t, W1_fp16, H, 2 * p.intermediate_dim);

      const float *B1 = w.b_mlp1 + (size_t)expert_index * (2 * p.intermediate_dim);
      add_bias_gpu(s.mlp1_out, B1, 2 * p.intermediate_dim);

      // split into gate/up (strided memcopy on device is okay via kernels, but here do 2 gemv-free copies)
      const int BS = 256, GS = (p.intermediate_dim + BS - 1) / BS;
      hipLaunchKernelGGL(split_gate_up, dim3(GS), dim3(BS), 0, 0,
                        s.mlp1_out, s.gate, s.up, p.intermediate_dim);

      // SwiGLU + clamp → gate_up
      swiglu_gpu(s.gate, s.up, s.gate_up, p.intermediate_dim, 1.702f, p.swiglu_limit);

      // MLP2: tb2 = W_mlp2[l,e] * gate_up + b_mlp2[l,e] → size hidden
      const float *W2 = w.w_mlp2 + (size_t)(l*p.n_experts + e) * H * p.intermediate_dim;
      const float *B2 = w.b_mlp2 + (size_t)(l*p.n_experts + e) * H;
      gemv_gpu(s.tb2, s.gate_up, W2, p.intermediate_dim, H);
      add_bias_gpu(s.tb2, B2, H);

      // e_agg += tb2 * wexp
      axpy_gpu(s.e_agg, s.tb2, wexp, H);
    }

    // residual: x += e_agg
    axpy_gpu(s.x, s.e_agg, 1.0f, H);
  } // end layer loop

  // final rmsnorm
  rmsnorm_gpu(s.x, s.x, w.rms_out_w, H);

  // logits = out * x
  gemv_gpu(s.logits, s.x, w.out, H, p.vocab_size);
  // HIP_CHECK(hipDeviceSynchronize());
  return s.logits;
}
