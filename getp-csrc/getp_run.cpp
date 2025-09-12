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

static void convert_fp32_to_fp16_host(const float *src, __half *dst, size_t N) {
    for (size_t i = 0; i < N; i++) {
        dst[i] = __float2half(src[i]);
        // if (i < 1000) printf("Converting fp32 to fp16: %.32f -> %.32f\n", src[i], (float)dst[i]);
    }
}

__half *d_w_mlp1_fp16 = nullptr;
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
  {
    size_t elems_w_mlp1 =
        (size_t)n_layers * (size_t)n_experts * (size_t)2 *
        (size_t)cfg->intermediate_dim * (size_t)cfg->hidden_dim;

    const float *host_w_mlp1 = ptr; // FP32 mmap buffer

    // Allocate temporary host FP16 buffer
    __half *h_w_mlp1_fp16 = (__half*)malloc(elems_w_mlp1 * sizeof(__half));
    if (!h_w_mlp1_fp16) {
        fprintf(stderr, "OOM: cannot allocate host FP16 buffer for w_mlp1\n");
        exit(1);
    }

    // Convert FP32 -> FP16 on host
    convert_fp32_to_fp16_host(host_w_mlp1, h_w_mlp1_fp16, elems_w_mlp1);

    // Allocate device buffer in FP16
    // __half *d_w_mlp1_fp16 = nullptr;
    HIP_CHECK(hipMalloc(&d_w_mlp1_fp16, elems_w_mlp1 * sizeof(__half)));

    // Copy host → device
    HIP_CHECK(hipMemcpy(d_w_mlp1_fp16, h_w_mlp1_fp16,
                        elems_w_mlp1 * sizeof(__half),
                        hipMemcpyHostToDevice));

    free(h_w_mlp1_fp16); // free temp buffer

    // Store in TransformerWeights (keep struct unchanged)
    // w->w_mlp1 = reinterpret_cast<float*>(d_w_mlp1_fp16);

    // Advance ptr by FP32 count (so following weights parse correctly)
    ptr += elems_w_mlp1;
  }
  to_device(&w->b_mlp1, ptr, 1ll * n_layers * n_experts * 2 * cfg->intermediate_dim * sizeof(float));
  ptr += 1ll * n_layers * n_experts * 2 * cfg->intermediate_dim;
  to_device(&w->w_mlp2, ptr,
            1ll * n_layers * n_experts * cfg->hidden_dim * cfg->intermediate_dim *
            sizeof(float));
  ptr += 1ll * n_layers * n_experts * cfg->hidden_dim * cfg->intermediate_dim;
  to_device(&w->b_mlp2, ptr, 1ll * n_layers * n_experts * cfg->hidden_dim * sizeof(float));
  ptr += 1ll * n_layers * n_experts * cfg->hidden_dim;
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

// ------------------------------ I/O helpers ------------------------------

static void free_transformer_gpu(Transformer *T) {
  // if (T->data && T->data!=MAP_FAILED) munmap(T->data, T->file_size);
  // if (T->fd!=-1) close(T->fd);

  // free device weights/state
  TransformerWeights &g = T->weights;
  auto F=[&](float *&p){ if(p){ hipFree(p); p=nullptr; } };
  F(g.token_embedding_table); F(g.rms_attn_w); F(g.rms_ffn_w); F(g.w_qkv); F(g.w_o);
  F(g.b_qkv); F(g.b_o); F(g.attn_sinks); F(g.w_router); F(g.b_router);
  F(g.w_mlp1); F(g.w_mlp2); F(g.b_mlp1); F(g.b_mlp2); 
  F(g.rms_out_w); F(g.out);

  RunState &s = T->state;
  F(s.x); F(s.t); F(s.tb); F(s.tb2); F(s.router_score); if(s.topk_v) hipFree(s.topk_v);
  if(s.topk_i) hipFree(s.topk_i); F(s.mlp1_out); F(s.gate); F(s.up); F(s.gate_up); F(s.e_agg);
  F(s.qkv); F(s.q); F(s.att); F(s.logits); F(s.key_cache); F(s.value_cache);
  if (s.mask) hipFree(s.mask);
}

static void build_transformer_gpu(Transformer *T) {
  // T->fd = -1; T->data = nullptr; T->file_size = 0;
  // hipSetDevice(0); // MI250 GCD0
  float *weights_ptr = T->data + sizeof(T->config) / sizeof(float);
  memory_map_weights_gpu(&T->weights, &T->config, weights_ptr);
  malloc_state_gpu(T);
}

void build_rope_tables(const Config& p, RopeTables &rt) {
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
    // HIP_CHECK(hipDeviceSynchronize());
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
    // hipSetDevice(0);
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
