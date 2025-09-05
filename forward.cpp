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
#include "hip/softmax.cpp"
#include "hip/matmul.cpp"
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

static void topk_gpu(float *h_topk_v, int *h_topk_i, const float *h_scores, int n, int k) {
  // Allocate host buffers
  // float *h_scores = (float*)malloc(n * sizeof(float));
  // float *h_topk_v = (float*)malloc(k * sizeof(float));
  // int   *h_topk_i = (int*)malloc(k * sizeof(int));

  // Copy scores from device to host
  // HIP_CHECK(hipMemcpy(h_scores, scores, n * sizeof(float), hipMemcpyDeviceToHost));

  // Top-k selection (same as host version)
  for (int i = 0; i < k; i++) { h_topk_v[i] = -INFINITY; h_topk_i[i] = -1; }
  for (int e = 0; e < n; e++) {
    float v = h_scores[e];
    int slot = -1;
    for (int i = 0; i < k; i++) if (v > h_topk_v[i]) { slot = i; break; }
    if (slot >= 0) {
      for (int j = k - 1; j > slot; j--) { h_topk_v[j] = h_topk_v[j - 1]; h_topk_i[j] = h_topk_i[j - 1]; }
      h_topk_v[slot] = v; h_topk_i[slot] = e;
    }
  }

  // Softmax normalize top-k (in-place)
  float mx = h_topk_v[0];
  for (int i = 1; i < k; i++) if (h_topk_v[i] > mx) mx = h_topk_v[i];
  double sum = 0.0;
  for (int i = 0; i < k; i++) { h_topk_v[i] = expf(h_topk_v[i] - mx); sum += h_topk_v[i]; }
  for (int i = 0; i < k; i++) h_topk_v[i] /= (float)sum;

  // Copy results back to device
  // HIP_CHECK(hipMemcpy(topk_v, h_topk_v, k * sizeof(float), hipMemcpyHostToDevice));
  // HIP_CHECK(hipMemcpy(topk_i, h_topk_i, k * sizeof(int), hipMemcpyHostToDevice));

  // free(h_scores);
  // free(h_topk_v);
  // free(h_topk_i);
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
    float *k_buf = T->state.key_cache + (size_t)l*p.seq_len*kv_dim + (size_t)pos*kv_dim;
    float *v_buf = T->state.value_cache + (size_t)l*p.seq_len*kv_dim + (size_t)pos*kv_dim;
    split_qkv_gpu(s.qkv, s.q, k_buf, v_buf, D, Hq, Hkv);

    // --- RoPE for q and k(pos)
    // compute cos/sin on host (cheap) then upload temporary
    int half = D/2;
    float *hcos = (float*)malloc(half*sizeof(float));
    float *hsin = (float*)malloc(half*sizeof(float));
    compute_cos_sin_host(pos, p.rope_theta, D, p.rope_scaling_factor,
                         p.initial_context_length, 32.0f, 1.0f, hcos, hsin);
    float *dcos, *dsin;
    HIP_CHECK(hipMalloc((void**)&dcos, half*sizeof(float)));
    HIP_CHECK(hipMalloc((void**)&dsin, half*sizeof(float)));
    HIP_CHECK(hipMemcpy(dcos, hcos, half*sizeof(float), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dsin, hsin, half*sizeof(float), hipMemcpyHostToDevice));
    free(hcos); free(hsin);

    rope_gpu(s.q, dcos, dsin, Hq, D);
    rope_gpu(k_buf, dcos, dsin, Hkv, D);
    HIP_CHECK(hipFree(dcos)); HIP_CHECK(hipFree(dsin));

    // --- Attention scores for all heads vs 0..pos
    float *k_layer_cache = T->state.key_cache + (size_t)l*p.seq_len*kv_dim;
    float *v_layer_cache = T->state.value_cache + (size_t)l*p.seq_len*kv_dim;
    attn_scores_gpu(s.q, k_layer_cache, (p.sliding_window>0 && (l % 2 == 0))? s.mask: nullptr, s.att,
                    D, kv_mul, p.seq_len, pos, kv_dim, Hq, (l%2==0)?p.sliding_window:0);

    // write sink score at index pos+1 (per head) from attn_sinks
    // We'll do it from host by staging a small vector:
    // step 1: copy att row for each head (pos+2 length) after kernel
    // simpler: small kernelless copy from host:
    float *h_sinks = (float*)malloc(Hq*sizeof(float));
    HIP_CHECK(hipMemcpy(h_sinks, w.attn_sinks + (size_t)l*Hq, Hq*sizeof(float), hipMemcpyDeviceToHost));
    // we just append sink value at [pos+1]
    for (int h=0; h<Hq; ++h) {
      float sinkv = h_sinks[h];
      HIP_CHECK(hipMemcpy(s.att + h*(p.seq_len+1) + (pos+1),
                          &sinkv, sizeof(float), hipMemcpyHostToDevice));
    }
    free(h_sinks);

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
    HIP_CHECK(hipDeviceSynchronize()); // ensure router on device finished

    // Bring router to host → top-k selection on CPU (simple, correct)
    float *h_router = (float*)malloc(p.n_experts*sizeof(float));
    float *h_topk_v = (float*)malloc(p.experts_per_token*sizeof(float));
    int   *h_topk_i = (int*)  malloc(p.experts_per_token*sizeof(int));
    HIP_CHECK(hipMemcpy(h_router, s.router_score, p.n_experts*sizeof(float), hipMemcpyDeviceToHost));
    topk_gpu(h_topk_v, h_topk_i, h_router, p.n_experts, p.experts_per_token);
    HIP_CHECK(hipMemcpy(s.topk_v, h_topk_v, p.experts_per_token*sizeof(float), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(s.topk_i, h_topk_i, p.experts_per_token*sizeof(int),   hipMemcpyHostToDevice));
    free(h_router);

    // e_agg = 0
    {
      const int BS=256, GS=(H+BS-1)/BS;
      hipLaunchKernelGGL(k_set, dim3(GS), dim3(BS), 0, 0, s.e_agg, 0.f, H);
    }

    // Host copies of topk (we had them as h_topk_v/h_topk_i earlier)
    // HIP_CHECK(hipMemcpy(h_topk_v, s.topk_v, p.experts_per_token*sizeof(float), hipMemcpyDeviceToHost));
    // HIP_CHECK(hipMemcpy(h_topk_i, s.topk_i, p.experts_per_token*sizeof(int),   hipMemcpyDeviceToHost));

    if (MOE_NGPUS <= 1) {
      // single-device path (original)
      for (int idx=0; idx<p.experts_per_token; ++idx) {
        int e = h_topk_i[idx];
        float wexp = h_topk_v[idx];

        const float *W1 = w.w_mlp1 + (size_t)(l*p.n_experts + e) * (2*p.intermediate_dim) * H;
        const float *B1 = w.b_mlp1 + (size_t)(l*p.n_experts + e) * (2*p.intermediate_dim);
        gemv_gpu(s.mlp1_out, s.t, W1, H, 2*p.intermediate_dim);
        add_bias_gpu(s.mlp1_out, B1, 2*p.intermediate_dim);

        HIP_CHECK(hipMemcpy2D(s.gate, sizeof(float),
                              s.mlp1_out, sizeof(float)*2,
                              sizeof(float), p.intermediate_dim, hipMemcpyDeviceToDevice));
        HIP_CHECK(hipMemcpy2D(s.up, sizeof(float),
                              s.mlp1_out+1, sizeof(float)*2,
                              sizeof(float), p.intermediate_dim, hipMemcpyDeviceToDevice));

        swiglu_gpu(s.gate, s.up, s.gate_up, p.intermediate_dim, 1.702f, p.swiglu_limit);

        const float *W2 = w.w_mlp2 + (size_t)(l*p.n_experts + e) * H * p.intermediate_dim;
        const float *B2 = w.b_mlp2 + (size_t)(l*p.n_experts + e) * H;
        gemv_gpu(s.tb2, s.gate_up, W2, p.intermediate_dim, H);
        add_bias_gpu(s.tb2, B2, H);

        axpy_gpu(s.e_agg, s.tb2, wexp, H);
      }
    } else {
      // multi-device expert-parallel path:
      // 1) stage s.t (input) to host once, then copy to each device's RunState.t and topk arrays
      float *h_t = (float*)malloc(H * sizeof(float));
      HIP_CHECK(hipMemcpy(h_t, s.t, H * sizeof(float), hipMemcpyDeviceToHost));

      // For each device, copy t and topk arrays into that device's RunState
      for (int d=0; d<MOE_NGPUS; ++d) {
        HIP_CHECK(hipSetDevice(d));
        // copy t
        HIP_CHECK(hipMemcpy(MOE_dev_state[d].t, h_t, H * sizeof(float), hipMemcpyHostToDevice));
        // copy topk arrays (common to all devices)
        HIP_CHECK(hipMemcpy(MOE_dev_state[d].topk_v, h_topk_v, p.experts_per_token*sizeof(float), hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(MOE_dev_state[d].topk_i, h_topk_i, p.experts_per_token*sizeof(int),   hipMemcpyHostToDevice));

        // zero per-device e_agg
        const int BS=256, GS=(H+BS-1)/BS;
        hipLaunchKernelGGL(k_set, dim3(GS), dim3(BS), 0, 0, MOE_dev_state[d].e_agg, 0.f, H);
      }

      free(h_t);

      // 2) compute experts assigned to each device in parallel (each device processes only its local experts)
      // #pragma omp parallel for num_threads(MOE_NGPUS)
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
          int e = h_topk_i[idx];
          float wexp = h_topk_v[idx];
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
          HIP_CHECK(hipMemcpy2D(ds.gate, sizeof(float),
                                ds.mlp1_out, sizeof(float)*2,
                                sizeof(float), p.intermediate_dim, hipMemcpyDeviceToDevice));
          HIP_CHECK(hipMemcpy2D(ds.up, sizeof(float),
                                ds.mlp1_out+1, sizeof(float)*2,
                                sizeof(float), p.intermediate_dim, hipMemcpyDeviceToDevice));

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

    free(h_topk_v);
    free(h_topk_i);

    // residual: x += e_agg
    axpy_gpu(s.x, s.e_agg, 1.0f, H);
  } // end layer loop

  // final rmsnorm
  rmsnorm_gpu(s.x, s.x, w.rms_out_w, H);

  // logits = out * x
  gemv_gpu(s.logits, s.x, w.out, H, p.vocab_size);
  HIP_CHECK(hipDeviceSynchronize());
  return s.logits;
}