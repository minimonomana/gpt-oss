// TODO: Modify this file to optimize end-to-end throughput
#include "getp_eval.cpp"
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

#if defined(_WIN32)
#include "win.h"
#else
#include <sys/mman.h>
#include <unistd.h>
#endif

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <omp.h>

#include "../tokenizer.hpp"

#include "../forward.cpp"
#include "../sample.cpp"

#ifndef GETP_RUN
#define GETP_RUN

// ------------------------------- Helpers ---------------------------------

#define HIP_CHECK(cmd) do { \
  hipError_t e = (cmd);     \
  if (e != hipSuccess) {    \
    fprintf(stderr, "HIP error %d (%s) at %s:%d\n", \
            (int)e, hipGetErrorString(e), __FILE__, __LINE__); \
    exit(1); \
  } \
} while(0)

// ------------------------ GPU setup / allocations ------------------------

static void to_device(float **dptr, const float *hptr, size_t nbytes) {
  HIP_CHECK(hipMalloc((void**)dptr, nbytes));
  HIP_CHECK(hipMemcpy(*dptr, hptr, nbytes, hipMemcpyHostToDevice));
}

static void alloc_device(float **dptr, size_t nbytes, float fill=0.f, bool set=false) {
  HIP_CHECK(hipMalloc((void**)dptr, nbytes));
  if (set) {
    int n = (int)(nbytes / sizeof(float));
    int bs = 256, gs = (n + bs - 1) / bs;
    hipLaunchKernelGGL(k_set, dim3(gs), dim3(bs), 0, 0, *dptr, fill, n);
    HIP_CHECK(hipDeviceSynchronize());
  }
}

// -------------------------- File mapping (host) --------------------------

void memory_map_weights_gpu(TransformerWeights *w, Config *cfg, float *ptr) {
  int head_dim = cfg->head_dim;
  int n_layers = cfg->n_layers;
  int n_experts = cfg->n_experts;

  to_device(&w->token_embedding_table, ptr, 1ll*cfg->vocab_size*cfg->hidden_dim*sizeof(float));
  ptr += 1ll * cfg->vocab_size * cfg->hidden_dim;
  to_device(&w->out, ptr, 1ll*cfg->vocab_size*cfg->hidden_dim*sizeof(float));
  ptr += 1ll * cfg->vocab_size * cfg->hidden_dim;
  to_device(&w->rms_attn_w, ptr, 1ll * n_layers * cfg->hidden_dim * sizeof(float));
  ptr += 1ll * n_layers * cfg->hidden_dim;
  to_device(&w->rms_ffn_w, ptr, 1ll * n_layers * cfg->hidden_dim * sizeof(float));
  ptr += 1ll * n_layers * cfg->hidden_dim;
  to_device(&w->rms_out_w, ptr, 1ll * cfg->hidden_dim * sizeof(float));
  ptr += 1ll * cfg->hidden_dim;
  // hey it's qkvqkv, not qqkkvv
  to_device(&w->w_qkv, ptr,
            1ll * n_layers * cfg->hidden_dim *
            (head_dim * cfg->n_attn_heads + 2 * head_dim * cfg->n_kv_heads) *
            sizeof(float));
  ptr += 1ll * n_layers * cfg->hidden_dim *
         (head_dim * cfg->n_attn_heads + 2 * head_dim * cfg->n_kv_heads);
  to_device(&w->b_qkv, ptr,
            1ll * n_layers * (head_dim * cfg->n_attn_heads + 2 * head_dim * cfg->n_kv_heads) *
            sizeof(float));
  ptr += 1ll * n_layers *
         (head_dim * cfg->n_attn_heads + 2 * head_dim * cfg->n_kv_heads);
  to_device(&w->w_o, ptr,
            1ll * n_layers * (head_dim * cfg->n_attn_heads) * cfg->hidden_dim *
            sizeof(float));
  ptr += 1ll * n_layers * (head_dim * cfg->n_attn_heads) * cfg->hidden_dim;
  to_device(&w->b_o, ptr, 1ll * n_layers * cfg->hidden_dim * sizeof(float));
  ptr += 1ll * n_layers * cfg->hidden_dim;
  to_device(&w->attn_sinks, ptr, 1ll * n_layers * cfg->n_attn_heads * sizeof(float));
  ptr += 1ll * n_layers * cfg->n_attn_heads;
  to_device(&w->w_router, ptr, 1ll * n_layers * cfg->hidden_dim * n_experts * sizeof(float));
  ptr += 1ll * n_layers * cfg->hidden_dim * n_experts;
  to_device(&w->b_router, ptr, 1ll * n_layers * n_experts * sizeof(float));
  ptr += 1ll * n_layers * n_experts;
  // hey it's gate_upgate_up, not gategateupup
  // to_device(&w->w_mlp1, ptr,
  //           1ll * n_layers * n_experts * cfg->hidden_dim * 2 * cfg->intermediate_dim *
  //           sizeof(float));
  // ptr +=
  //     1ll * n_layers * n_experts * 2 * cfg->intermediate_dim * cfg->hidden_dim;
  // to_device(&w->b_mlp1, ptr, 1ll * n_layers * n_experts * 2 * cfg->intermediate_dim * sizeof(float));
  // ptr += 1ll * n_layers * n_experts * 2 * cfg->intermediate_dim;
  // to_device(&w->w_mlp2, ptr,
  //           1ll * n_layers * n_experts * cfg->hidden_dim * cfg->intermediate_dim *
  //           sizeof(float));
  // ptr += 1ll * n_layers * n_experts * cfg->hidden_dim * cfg->intermediate_dim;
  // to_device(&w->b_mlp2, ptr, 1ll * n_layers * n_experts * cfg->hidden_dim * sizeof(float));
  // ptr += 1ll * n_layers * n_experts * cfg->hidden_dim;
}

static void malloc_state_gpu(Transformer *T) {
  const Config &c = T->config;
  RunState &s = T->state;

  alloc_device(&s.x,        c.hidden_dim*sizeof(float), 0.f, true);
  alloc_device(&s.t,        c.hidden_dim*sizeof(float), 0.f, true);
  alloc_device(&s.tb,       c.head_dim*c.n_attn_heads*sizeof(float), 0.f, true);
  alloc_device(&s.tb2,      c.hidden_dim*sizeof(float), 0.f, true);
  alloc_device(&s.router_score,   c.n_experts*sizeof(float), 0.f, true);
  HIP_CHECK(hipMalloc((void**)&s.topk_v, c.experts_per_token*sizeof(float)));
  HIP_CHECK(hipMalloc((void**)&s.topk_i, c.experts_per_token*sizeof(int)));
  alloc_device(&s.mlp1_out, 2*c.intermediate_dim*sizeof(float), 0.f, true);
  alloc_device(&s.gate,     c.intermediate_dim*sizeof(float), 0.f, true);
  alloc_device(&s.up,       c.intermediate_dim*sizeof(float), 0.f, true);
  alloc_device(&s.gate_up,  c.intermediate_dim*sizeof(float), 0.f, true);
  alloc_device(&s.e_agg,    c.hidden_dim*sizeof(float), 0.f, true);

  int qkv_tot = c.head_dim*(c.n_attn_heads + 2*c.n_kv_heads);
  alloc_device(&s.qkv,    qkv_tot*sizeof(float), 0.f, true);
  alloc_device(&s.q,      c.head_dim*c.n_attn_heads*sizeof(float), 0.f, true);
  // k_cur/v_cur views are offsets into caches (no alloc here)
  alloc_device(&s.att,    (c.n_attn_heads*(c.seq_len+1))*sizeof(float), 0.f, true);
  alloc_device(&s.logits, c.vocab_size*sizeof(float), 0.f, true);

  int kv_dim = c.head_dim * c.n_kv_heads;
  size_t cache_elems = 1ll*c.n_layers*c.seq_len*kv_dim;
  alloc_device(&s.key_cache,   cache_elems*sizeof(float), 0.f, true);
  alloc_device(&s.value_cache, cache_elems*sizeof(float), 0.f, true);

  if (c.sliding_window > 0) {
    alloc_device(&s.mask, 1ll*c.seq_len*c.seq_len*sizeof(float), 0.f, true);
    // host-init mask once then copy
    float *hmask = (float*)malloc(1ll*c.seq_len*c.seq_len*sizeof(float));
    for (int i=0;i<c.seq_len;i++) for (int j=0;j<c.seq_len;j++) {
      float v = 0.f;
      if (c.sliding_window > 0 && i - j >= c.sliding_window) v = -INFINITY;
      hmask[i*c.seq_len + j] = v;
    }
    HIP_CHECK(hipMemcpy(s.mask, hmask, 1ll*c.seq_len*c.seq_len*sizeof(float), hipMemcpyHostToDevice));
    free(hmask);
  } else {
    s.mask = nullptr;
  }
}

// ---------- Multi-GPU MoE infra (add to getp_run.cpp) ----------
// Place after your existing helpers/kernels, before build_transformer_gpu/forward_gpu.

int MOE_NGPUS = 0;           // number of devices used for expert parallelism
int MOE_GROUP_SIZE = 0;      // experts per device (n_experts / MOE_NGPUS)

// per-device pointers for MoE shards and per-device RunState
float **MOE_dev_w_mlp1 = nullptr;
float **MOE_dev_w_mlp2 = nullptr;
float **MOE_dev_b_mlp1 = nullptr;
float **MOE_dev_b_mlp2 = nullptr;
RunState *MOE_dev_state = nullptr;      // host-side array of RunState (each entry holds device pointers)
float **MOE_partial_on_dev0 = nullptr;   // device-0 pointers for partial results (one per device)

// Helper: allocate RunState fields on a given device (uses alloc_device helper already in file)
// note: call hipSetDevice(dev) before calling this function
static void alloc_runstate_on_device(RunState &s, const Config &c) {
  alloc_device(&s.x,        c.hidden_dim*sizeof(float), 0.f, true);
  alloc_device(&s.t,        c.hidden_dim*sizeof(float), 0.f, true);
  alloc_device(&s.tb,       c.head_dim*c.n_attn_heads*sizeof(float), 0.f, true);
  alloc_device(&s.tb2,      c.hidden_dim*sizeof(float), 0.f, true);
  alloc_device(&s.router_score,   c.n_experts*sizeof(float), 0.f, true);
  HIP_CHECK(hipMalloc((void**)&s.topk_v, c.experts_per_token*sizeof(float)));
  HIP_CHECK(hipMalloc((void**)&s.topk_i, c.experts_per_token*sizeof(int)));
  alloc_device(&s.mlp1_out, 2*c.intermediate_dim*sizeof(float), 0.f, true);
  alloc_device(&s.gate,     c.intermediate_dim*sizeof(float), 0.f, true);
  alloc_device(&s.up,       c.intermediate_dim*sizeof(float), 0.f, true);
  alloc_device(&s.gate_up,  c.intermediate_dim*sizeof(float), 0.f, true);
  alloc_device(&s.e_agg,    c.hidden_dim*sizeof(float), 0.f, true);

  int qkv_tot = c.head_dim*(c.n_attn_heads + 2*c.n_kv_heads);
  alloc_device(&s.qkv,    qkv_tot*sizeof(float), 0.f, true);
  alloc_device(&s.q,      c.head_dim*c.n_attn_heads*sizeof(float), 0.f, true);
  alloc_device(&s.att,    (c.n_attn_heads*(c.seq_len+1))*sizeof(float), 0.f, true);
  alloc_device(&s.logits, c.vocab_size*sizeof(float), 0.f, true);

  int kv_dim = c.head_dim * c.n_kv_heads;
  size_t cache_elems = 1ll*c.n_layers*c.seq_len*kv_dim;
  alloc_device(&s.key_cache,   cache_elems*sizeof(float), 0.f, true);
  alloc_device(&s.value_cache, cache_elems*sizeof(float), 0.f, true);

  if (c.sliding_window > 0) {
    alloc_device(&s.mask, 1ll*c.seq_len*c.seq_len*sizeof(float), 0.f, true);
    // initialize mask same as device 0 code: caller should copy host mask into this device mask if needed.
    float *hmask = (float*)malloc(1ll*c.seq_len*c.seq_len*sizeof(float));
    for (int i=0;i<c.seq_len;i++) for (int j=0;j<c.seq_len;j++) {
      float v = 0.f;
      if (c.sliding_window > 0 && i - j >= c.sliding_window) v = -INFINITY;
      hmask[i*c.seq_len + j] = v;
    }
    HIP_CHECK(hipMemcpy(s.mask, hmask, 1ll*c.seq_len*c.seq_len*sizeof(float), hipMemcpyHostToDevice));
    free(hmask);
  } else {
    s.mask = nullptr;
  }
}

// Helper: free RunState fields (call with hipSetDevice(dev) to free device-side memory)
static void free_runstate_on_device(RunState &s) {
  auto F = [&](float *&p){ if (p){ hipFree(p); p=nullptr; } };
  if (s.topk_v) hipFree(s.topk_v);
  if (s.topk_i) hipFree(s.topk_i);
  F(s.x); F(s.t); F(s.tb); F(s.tb2); F(s.router_score); F(s.mlp1_out);
  F(s.gate); F(s.up); F(s.gate_up); F(s.e_agg);
  F(s.qkv); F(s.q); F(s.att); F(s.logits); F(s.key_cache); F(s.value_cache);
  if (s.mask) { hipFree(s.mask); s.mask = nullptr; }
}

// Initialize MoE multi-GPU, scatter MoE weights from host-mapped checkpoint and allocate per-device state.
// Call this after load_checkpoint_gpu(...) and after malloc_state_gpu(T) so T->data (host mmap) exists.
static void init_moe_gpu(Transformer *T, int requested_ngpus = 0) {
  const Config &c = T->config;
  int available = 0;
  HIP_CHECK(hipGetDeviceCount(&available));
  // pick how many GPUs to use for expert parallelism
  int ng = (requested_ngpus > 0 && requested_ngpus <= available) ? requested_ngpus : available;
  if (ng <= 1) {
    MOE_NGPUS = 1;
    MOE_GROUP_SIZE = c.n_experts;
    return;
  }
  MOE_NGPUS = ng;
  MOE_GROUP_SIZE = c.n_experts / MOE_NGPUS;
  if (MOE_GROUP_SIZE * MOE_NGPUS != c.n_experts) {
    fprintf(stderr, "MOE partitioning requires n_experts divisible by ngpus\n");
    exit(1);
  }

  // allocate arrays
  MOE_dev_w_mlp1 = (float**)malloc(sizeof(float*) * MOE_NGPUS);
  MOE_dev_w_mlp2 = (float**)malloc(sizeof(float*) * MOE_NGPUS);
  MOE_dev_b_mlp1 = (float**)malloc(sizeof(float*) * MOE_NGPUS);
  MOE_dev_b_mlp2 = (float**)malloc(sizeof(float*) * MOE_NGPUS);
  MOE_partial_on_dev0 = (float**)malloc(sizeof(float*) * MOE_NGPUS);
  MOE_dev_state = (RunState*)malloc(sizeof(RunState) * MOE_NGPUS);
  memset(MOE_dev_state, 0, sizeof(RunState) * MOE_NGPUS);

  // compute host base pointer for MoE weights inside the mmap'd file:
  float *host_base = nullptr;
  if (T->data == nullptr) {
    fprintf(stderr, "init_moe_gpu: T->data (host mmap) is null - cannot scatter weights\n");
    exit(1);
  }
  host_base = T->data + sizeof(Config)/sizeof(float);
  float *ptr = host_base;

  // Walk offsets in same order as memory_map_weights_gpu to reach w_mlp1,b_mlp1,w_mlp2,b_mlp2
  ptr += 1ll * c.vocab_size * c.hidden_dim; // token_embedding_table
  ptr += 1ll * c.vocab_size * c.hidden_dim; // out
  ptr += 1ll * c.n_layers * c.hidden_dim;   // rms_attn_w
  ptr += 1ll * c.n_layers * c.hidden_dim;   // rms_ffn_w
  ptr += 1ll * c.hidden_dim;                // rms_out_w
  ptr += 1ll * c.n_layers * c.hidden_dim * (c.head_dim * c.n_attn_heads + 2 * c.head_dim * c.n_kv_heads); // w_qkv
  ptr += 1ll * c.n_layers * (c.head_dim * c.n_attn_heads + 2 * c.head_dim * c.n_kv_heads); // b_qkv
  ptr += 1ll * c.n_layers * (c.head_dim * c.n_attn_heads) * c.hidden_dim; // w_o
  ptr += 1ll * c.n_layers * c.hidden_dim; // b_o
  ptr += 1ll * c.n_layers * c.n_attn_heads; // attn_sinks
  ptr += 1ll * c.n_layers * c.hidden_dim * c.n_experts; // w_router
  ptr += 1ll * c.n_layers * c.n_experts; // b_router

  // Now ptr points to start of w_mlp1
  float *host_w_mlp1 = ptr;
  size_t mlp1_per_expert = (size_t)2 * c.intermediate_dim * c.hidden_dim;
  ptr += 1ll * c.n_layers * c.n_experts * mlp1_per_expert;

  float *host_b_mlp1 = ptr;
  ptr += 1ll * c.n_layers * c.n_experts * (size_t)(2 * c.intermediate_dim);

  float *host_w_mlp2 = ptr;
  size_t mlp2_per_expert = (size_t)c.hidden_dim * c.intermediate_dim;
  ptr += 1ll * c.n_layers * c.n_experts * mlp2_per_expert;

  float *host_b_mlp2 = ptr;
  ptr += 1ll * c.n_layers * c.n_experts * (size_t)c.hidden_dim;

  // Now scatter per-device
  size_t dev_mlp1_elems_per_layer = (size_t)MOE_GROUP_SIZE * mlp1_per_expert;
  size_t dev_mlp2_elems_per_layer = (size_t)MOE_GROUP_SIZE * mlp2_per_expert;
  size_t dev_b1_elems_per_layer = (size_t)MOE_GROUP_SIZE * (2 * c.intermediate_dim);
  size_t dev_b2_elems_per_layer = (size_t)MOE_GROUP_SIZE * c.hidden_dim;

  for (int d=0; d<MOE_NGPUS; ++d) {
    HIP_CHECK(hipSetDevice(d));
    // allocate device-local shard buffers sized for all layers
    size_t dev_mlp1_total = (size_t)c.n_layers * dev_mlp1_elems_per_layer;
    size_t dev_b1_total   = (size_t)c.n_layers * dev_b1_elems_per_layer;
    size_t dev_mlp2_total = (size_t)c.n_layers * dev_mlp2_elems_per_layer;
    size_t dev_b2_total   = (size_t)c.n_layers * dev_b2_elems_per_layer;

    HIP_CHECK( hipMalloc((void**)&MOE_dev_w_mlp1[d], dev_mlp1_total * sizeof(float)) );
    HIP_CHECK( hipMalloc((void**)&MOE_dev_b_mlp1[d], dev_b1_total * sizeof(float)) );
    HIP_CHECK( hipMalloc((void**)&MOE_dev_w_mlp2[d], dev_mlp2_total * sizeof(float)) );
    HIP_CHECK( hipMalloc((void**)&MOE_dev_b_mlp2[d], dev_b2_total * sizeof(float)) );

    // copy per-layer slices from host into this device's contiguous buffer
    for (int l=0; l<c.n_layers; ++l) {
      float *h_slice_w1 = host_w_mlp1 + (size_t)l * c.n_experts * mlp1_per_expert
                               + (size_t)d * MOE_GROUP_SIZE * mlp1_per_expert;
      float *dst_w1 = MOE_dev_w_mlp1[d] + (size_t)l * dev_mlp1_elems_per_layer;
      HIP_CHECK( hipMemcpy(dst_w1, h_slice_w1, dev_mlp1_elems_per_layer * sizeof(float), hipMemcpyHostToDevice) );

      float *h_slice_b1 = host_b_mlp1 + (size_t)l * c.n_experts * (2 * c.intermediate_dim)
                               + (size_t)d * MOE_GROUP_SIZE * (2 * c.intermediate_dim);
      float *dst_b1 = MOE_dev_b_mlp1[d] + (size_t)l * dev_b1_elems_per_layer;
      HIP_CHECK( hipMemcpy(dst_b1, h_slice_b1, dev_b1_elems_per_layer * sizeof(float), hipMemcpyHostToDevice) );

      float *h_slice_w2 = host_w_mlp2 + (size_t)l * c.n_experts * mlp2_per_expert
                               + (size_t)d * MOE_GROUP_SIZE * mlp2_per_expert;
      float *dst_w2 = MOE_dev_w_mlp2[d] + (size_t)l * dev_mlp2_elems_per_layer;
      HIP_CHECK( hipMemcpy(dst_w2, h_slice_w2, dev_mlp2_elems_per_layer * sizeof(float), hipMemcpyHostToDevice) );

      float *h_slice_b2 = host_b_mlp2 + (size_t)l * c.n_experts * c.hidden_dim
                               + (size_t)d * MOE_GROUP_SIZE * c.hidden_dim;
      float *dst_b2 = MOE_dev_b_mlp2[d] + (size_t)l * dev_b2_elems_per_layer;
      HIP_CHECK( hipMemcpy(dst_b2, h_slice_b2, dev_b2_elems_per_layer * sizeof(float), hipMemcpyHostToDevice) );
    }

    // allocate partial result buffer on device 0 (host-visible) for gathering; allocate on device 0
    HIP_CHECK(hipSetDevice(0));
    HIP_CHECK( hipMalloc((void**)&MOE_partial_on_dev0[d], c.hidden_dim * sizeof(float)) );

    // allocate per-device RunState
    HIP_CHECK(hipSetDevice(d));
    alloc_runstate_on_device(MOE_dev_state[d], c);
  }

  // note: we keep device0's full T->weights*(all) untouched; the MoE shards additionally exist on each device.
  // Optionally we could free the full-device copy of MoE weights on device0 to save memory if necessary.
}

// cleanup
static void free_moe_gpu(Transformer *T) {
  const Config &c = T->config;
  if (MOE_NGPUS <= 1) return;
  for (int d=0; d<MOE_NGPUS; ++d) {
    // free per-device shards
    HIP_CHECK( hipSetDevice(d) );
    if (MOE_dev_w_mlp1 && MOE_dev_w_mlp1[d]) hipFree(MOE_dev_w_mlp1[d]);
    if (MOE_dev_w_mlp2 && MOE_dev_w_mlp2[d]) hipFree(MOE_dev_w_mlp2[d]);
    if (MOE_dev_b_mlp1 && MOE_dev_b_mlp1[d]) hipFree(MOE_dev_b_mlp1[d]);
    if (MOE_dev_b_mlp2 && MOE_dev_b_mlp2[d]) hipFree(MOE_dev_b_mlp2[d]);
    // free per-device RunState
    free_runstate_on_device(MOE_dev_state[d]);
    // free partial buffers on device 0
    HIP_CHECK( hipSetDevice(0) );
    if (MOE_partial_on_dev0 && MOE_partial_on_dev0[d]) hipFree(MOE_partial_on_dev0[d]);
  }
  if (MOE_dev_w_mlp1) free(MOE_dev_w_mlp1);
  if (MOE_dev_w_mlp2) free(MOE_dev_w_mlp2);
  if (MOE_dev_b_mlp1) free(MOE_dev_b_mlp1);
  if (MOE_dev_b_mlp2) free(MOE_dev_b_mlp2);
  if (MOE_partial_on_dev0) free(MOE_partial_on_dev0);
  if (MOE_dev_state) free(MOE_dev_state);

  MOE_dev_w_mlp1 = MOE_dev_w_mlp2 = MOE_dev_b_mlp1 = MOE_dev_b_mlp2 = nullptr;
  MOE_partial_on_dev0 = nullptr;
  MOE_dev_state = nullptr;
  MOE_NGPUS = 0; MOE_GROUP_SIZE = 0;
}

// ------------------------------ I/O helpers ------------------------------

static void free_transformer_gpu(Transformer *T) {
  // if (T->data && T->data!=MAP_FAILED) munmap(T->data, T->file_size);
  // if (T->fd!=-1) close(T->fd);

  // free device weights/state
  free_moe_gpu(T); // free MoE multi-GPU infra if any
  TransformerWeights &g = T->weights;
  auto F=[&](float *&p){ if(p){ hipFree(p); p=nullptr; } };
  F(g.token_embedding_table); F(g.rms_attn_w); F(g.rms_ffn_w); F(g.w_qkv); F(g.w_o);
  F(g.b_qkv); F(g.b_o); F(g.attn_sinks); F(g.w_router); F(g.b_router);
  // F(g.w_mlp1); F(g.w_mlp2); F(g.b_mlp1); F(g.b_mlp2); 
  F(g.rms_out_w); F(g.out);

  // RunState &s = T->state;
  // F(s.x); F(s.t); F(s.tb); F(s.tb2); F(s.router_score); if(s.topk_v) hipFree(s.topk_v);
  // if(s.topk_i) hipFree(s.topk_i); F(s.mlp1_out); F(s.gate); F(s.up); F(s.gate_up); F(s.e_agg);
  // F(s.qkv); F(s.q); F(s.att); F(s.logits); F(s.key_cache); F(s.value_cache);
  // if (s.mask) hipFree(s.mask);
}

static void build_transformer_gpu(Transformer *T) {
  // T->fd = -1; T->data = nullptr; T->file_size = 0;
  // hipSetDevice(0); // MI250 GCD0
  float *weights_ptr = T->data + sizeof(T->config) / sizeof(float);
  memory_map_weights_gpu(&T->weights, &T->config, weights_ptr);
  malloc_state_gpu(T);
  init_moe_gpu(T, 2); // use 4 GPUs for MoE expert parallelism
}

void build_rope_tables(const Config& p, RopeTables &rt) {
  hipSetDevice(0);
    rt.max_seq = p.seq_len;
    rt.head_dim = p.head_dim;
    rt.half = p.head_dim / 2;

    size_t rope_size = (size_t)rt.max_seq * rt.half;
    HIP_CHECK(hipMalloc(&rt.dcos, rope_size * sizeof(float)));
    HIP_CHECK(hipMalloc(&rt.dsin, rope_size * sizeof(float)));

    dim3 block(256);
    dim3 grid((rope_size + block.x - 1) / block.x);
    k_precompute_rope<<<grid, block>>>(p.rope_theta, p.head_dim,
                                       p.rope_scaling_factor, p.initial_context_length,
                                       32.0f, 1.0f,
                                       rt.max_seq, rt.dcos, rt.dsin);
    HIP_CHECK(hipDeviceSynchronize());
}

RopeTables g_rope_tables;

void warm_up(Transformer *transformer, Tokenizer *tokenizer) {
  // Do not inference here
  // You should handle the warm-up process
  // TODO:
  // - Memory allocation
  // - Load model
  // - ...
  const char *tokenizer_path = "tokenizer.bin";

  build_transformer_gpu(transformer);
  build_rope_tables(transformer->config, g_rope_tables);
  // read_tokenizer(tokenizer, tokenizer_path, transformer->config.vocab_size);
}

void finish(Transformer *transformer, Tokenizer *tokenizer) {
  // Do not inference here
  // You should handle the finish process
  // TODO:
  // - Memory deallocation
  // - Unload model
  // - ...
  free_transformer_gpu(transformer);
  // free_tokenizer(tokenizer);
}

long long simple_getp_generate(Transformer *transformer, Tokenizer *tokenizer,
                               Sampler *sampler, const char *input_seq,
                               int *output_tokens, int steps) {
  // <|start|>: 200006
  // <|end|>: 200007
  // <|return|>: 200002
  // <|message|>: 200008
  // <|channel|>: 200005
  // <|constrain|>: 200003
  // <|endoftext|>: 199999

  // Inference here

  const char *empty_prompt = "";
  if (input_seq == NULL) {
    input_seq = empty_prompt;
  }

  // encode the (string) prompt into tokens sequence
  int num_prompt_tokens = 0;
  int *prompt_tokens = (int *)malloc((strlen(input_seq) + 3) *
                                     sizeof(int)); // +3 for '\0', ?BOS, ?EOS
  encode(tokenizer, input_seq, -1, -1, prompt_tokens, &num_prompt_tokens,
         transformer->config.initial_context_length);
  if (num_prompt_tokens < 1) {
    fprintf(stderr, "something is wrong, expected at least 1 prompt token\n");
    exit(EXIT_FAILURE);
  }

  // start the main loop
  int next;                     // will store the next token in the sequence
  int token = prompt_tokens[0]; // kick off with the first token in the prompt
  int pos = 0;                  // position in the sequence

  // print the very first token
  // should be removed
  const char *first_piece = decode_piece(tokenizer, 200006, token);
  safe_printf(first_piece);
  fflush(stdout);

  while (pos < steps) {

    // forward the transformer to get logits for the next token
    hipSetDevice(0);
    float *logits = forward_gpu(transformer, token, pos);

    // advance the state machine
    pos++;
    if (pos < num_prompt_tokens) {
      // if we are still processing the input prompt, force the next prompt
      // token
      next = prompt_tokens[pos];
    } else {
      // otherwise sample the next token from the logits
      next = sample_gpu(sampler, logits);
      // save the output token, it will be printed to file
      output_tokens[pos - num_prompt_tokens] = next;
    }

    // data-dependent terminating condition: the EOS (=199999 or =200002) token
    // delimits sequences
    if (next == 199999 || next == 200002) {
      break;
    }

    // print the token as string, decode it with the Tokenizer object
    // should be removed
    const char *piece = decode_piece(tokenizer, token, next);
    safe_printf(piece); // same as printf("%s", piece), but skips "unsafe" bytes
    fflush(stdout);

    token = next;
  }

  // should be removed
  printf("\n");

  // Marker for end of sequence
  output_tokens[pos - num_prompt_tokens + 1] = -1;

  free(prompt_tokens);

  return pos - num_prompt_tokens + 1;
}

long long inference(Transformer *transformer, Tokenizer *tokenizer,
                    Sampler *sampler, Requests *requests) {
  long long num_token_out = 0;
  for (int idx = 0; idx < requests->num_reqs; ++idx) {
    const char *input_seq = get_str_req_ptr(requests, idx);
    int *output_tokens = get_tok_gen_ptr(requests, idx);
    num_token_out +=
        simple_getp_generate(transformer, tokenizer, sampler, input_seq,
                             output_tokens, requests->max_seq_len);
  }
  return num_token_out;
}

#endif // GETP_RUN
