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

extern int MOE_NGPUS;
extern int MOE_GROUP_SIZE;
extern float **MOE_dev_w_mlp1;
extern float **MOE_dev_w_mlp2;
extern float **MOE_dev_b_mlp1;
extern float **MOE_dev_b_mlp2;
extern RunState *MOE_dev_state;
extern float **MOE_partial_on_dev0;
extern RopeTables g_rope_tables;

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
  hipSetDevice(0);
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

    if (MOE_NGPUS <= 1) {
      // single-device path (original)
      for (int idx=0; idx<p.experts_per_token; ++idx) {
        int e = s.topk_i[idx];
        float wexp = s.topk_v[idx];

        const float *W1 = w.w_mlp1 + (size_t)(l*p.n_experts + e) * (2*p.intermediate_dim) * H;
        const float *B1 = w.b_mlp1 + (size_t)(l*p.n_experts + e) * (2*p.intermediate_dim);
        gemv_gpu(s.mlp1_out, s.t, W1, H, 2*p.intermediate_dim);
        add_bias_gpu(s.mlp1_out, B1, 2*p.intermediate_dim);

        const int BS = 256, GS = (p.intermediate_dim + BS - 1) / BS;
        hipLaunchKernelGGL(split_gate_up, dim3(GS), dim3(BS), 0, 0,
                          s.mlp1_out, s.gate, s.up, p.intermediate_dim);

        swiglu_gpu(s.gate, s.up, s.gate_up, p.intermediate_dim, 1.702f, p.swiglu_limit);

        const float *W2 = w.w_mlp2 + (size_t)(l*p.n_experts + e) * H * p.intermediate_dim;
        const float *B2 = w.b_mlp2 + (size_t)(l*p.n_experts + e) * H;
        gemv_gpu(s.tb2, s.gate_up, W2, p.intermediate_dim, H);
        add_bias_gpu(s.tb2, B2, H);

        axpy_gpu(s.e_agg, s.tb2, wexp, H);
      }
    } else {
      // multi-device expert-parallel path:
      // For each device, copy t and topk arrays into that device's RunState
      for (int d=0; d<MOE_NGPUS; ++d) {
        HIP_CHECK(hipSetDevice(d));
        HIP_CHECK(hipMemcpyPeer(MOE_dev_state[d].t, d, s.t, 0, H * sizeof(float)));
        HIP_CHECK(hipMemcpyPeer(MOE_dev_state[d].topk_v, d, s.topk_v, 0, p.experts_per_token * sizeof(float)));
        HIP_CHECK(hipMemcpyPeer(MOE_dev_state[d].topk_i, d, s.topk_i, 0, p.experts_per_token * sizeof(int)));

        // zero per-device e_agg
        const int BS=256, GS=(H+BS-1)/BS;
        hipLaunchKernelGGL(k_set, dim3(GS), dim3(BS), 0, 0, MOE_dev_state[d].e_agg, 0.f, H);
      }

      // 2) compute experts assigned to each device in parallel (each device processes only its local experts)
      #pragma omp parallel for num_threads(MOE_NGPUS)
      for (int d=0; d<MOE_NGPUS; ++d) {
        HIP_CHECK(hipSetDevice(d));
        RunState &ds = MOE_dev_state[d];

        // local group base index
        int group_base = d * MOE_GROUP_SIZE;
        size_t mlp1_per = (size_t)2 * p.intermediate_dim * H;
        size_t mlp2_per = (size_t)H * p.intermediate_dim;
        size_t dev_mlp1_layer_stride = (size_t)MOE_GROUP_SIZE * mlp1_per;
        size_t dev_mlp2_layer_stride = (size_t)MOE_GROUP_SIZE * mlp2_per;
        size_t dev_b1_layer_stride   = (size_t)MOE_GROUP_SIZE * (2 * p.intermediate_dim);
        size_t dev_b2_layer_stride   = (size_t)MOE_GROUP_SIZE * H;

        // for each selected top-k entry, check whether it belongs to this device's group
        for (int idx=0; idx<p.experts_per_token; ++idx) {
          int e = ds.topk_i[idx];
          float wexp = ds.topk_v[idx];
          if (e < 0) continue;
          int owner = e / MOE_GROUP_SIZE;
          if (owner != d) continue;
          int local_idx = e - group_base;
          // pointers into device's shard
          const float *W1_local = MOE_dev_w_mlp1[d] + (size_t)l * dev_mlp1_layer_stride + (size_t)local_idx * mlp1_per;
          const float *B1_local = MOE_dev_b_mlp1[d] + (size_t)l * dev_b1_layer_stride + (size_t)local_idx * (2 * p.intermediate_dim);
          // mlp1: mlp1_out = W1 * ds.t + b1
          gemv_gpu(ds.mlp1_out, ds.t, W1_local, H, 2 * p.intermediate_dim);
          add_bias_gpu(ds.mlp1_out, B1_local, 2 * p.intermediate_dim);

          // split
          const int BS = 256, GS = (p.intermediate_dim + BS - 1) / BS;
          hipLaunchKernelGGL(split_gate_up, dim3(GS), dim3(BS), 0, 0,
                            ds.mlp1_out, ds.gate, ds.up, p.intermediate_dim);

          // SwiGLU
          swiglu_gpu(ds.gate, ds.up, ds.gate_up, p.intermediate_dim, 1.702f, p.swiglu_limit);

          // mlp2
          const float *W2_local = MOE_dev_w_mlp2[d] + (size_t)l * dev_mlp2_layer_stride + (size_t)local_idx * mlp2_per;
          const float *B2_local = MOE_dev_b_mlp2[d] + (size_t)l * dev_b2_layer_stride + (size_t)local_idx * H;
          gemv_gpu(ds.tb2, ds.gate_up, W2_local, p.intermediate_dim, H);
          add_bias_gpu(ds.tb2, B2_local, H);

          // accumulate into local device e_agg
          axpy_gpu(ds.e_agg, ds.tb2, wexp, H);
        } // end idx loop

        // after device computed its local e_agg, copy partial e_agg to device 0 partial buffer
        // copy device d: ds.e_agg --> device 0: MOE_partial_on_dev0[d]
        HIP_CHECK( hipMemcpyPeer(MOE_partial_on_dev0[d], 0, ds.e_agg, d, H * sizeof(float)) );
      } // end parallel for

      // 3) reduce partials on device 0: s.e_agg = sum_{d} partial_d
      // zero s.e_agg first (already zeroed above), now accumulate per-device
      for (int d=0; d<MOE_NGPUS; ++d) {
        // elementwise add MOE_partial_on_dev0[d] into s.e_agg on device 0
        HIP_CHECK(hipSetDevice(0));
        // use axpy kernel to do s.e_agg += partial
        axpy_gpu(s.e_agg, MOE_partial_on_dev0[d], 1.0f, H);
      }
    } // end multi-gpu MoE

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
