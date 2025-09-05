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

// #include "hip/softmax.cpp"

#define HIP_CHECK(cmd) do { \
  hipError_t e = (cmd);     \
  if (e != hipSuccess) {    \
    fprintf(stderr, "HIP error %d (%s) at %s:%d\n", \
            (int)e, hipGetErrorString(e), __FILE__, __LINE__); \
    exit(1); \
  } \
} while(0)

// Kernel to scale logits by temperature
__global__ void k_scale_logits(float *logits, float temp, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) logits[i] /= temp;
}

// GPU softmax for a single vector (use your kernel from hip/softmax.cpp)
static void softmax_gpu(float *x, int n) {
    // Launch the kernel for a single row
    softmax_rows_gpu(x, 1, n, n); // 1 head, row_len=n, stride=n
    HIP_CHECK(hipDeviceSynchronize());
}

// GPU argmax
static int sample_argmax_gpu(const float *p, int n) {
    float *h_p = (float*)malloc(n * sizeof(float));
    HIP_CHECK(hipMemcpy(h_p, p, n * sizeof(float), hipMemcpyDeviceToHost));
    int m = 0;
    float mv = h_p[0];
    for (int i = 1; i < n; i++) {
        if (h_p[i] > mv) { mv = h_p[i]; m = i; }
    }
    free(h_p);
    return m;
}

// GPU multinomial sampling
static int sample_mult_gpu(float *p, int n, float coin) {
    float *h_p = (float*)malloc(n * sizeof(float));
    HIP_CHECK(hipMemcpy(h_p, p, n * sizeof(float), hipMemcpyDeviceToHost));
    float cdf = 0.0f;
    for (int i = 0; i < n; i++) {
        cdf += h_p[i];
        if (coin < cdf) {
            free(h_p);
            return i;
        }
    }
    free(h_p);
    return n - 1;
}

static int sample_topp_gpu(float *p, int n, float topp, ProbIndex *probindex, float coin) {
    float *h_p = (float*)malloc(n * sizeof(float));
    HIP_CHECK(hipMemcpy(h_p, p, n * sizeof(float), hipMemcpyDeviceToHost));
    int n0 = 0;
    const float cutoff = (1.0f - topp) / (n - 1);
    for (int i = 0; i < n; i++) {
        if (h_p[i] >= cutoff) {
            probindex[n0].index = i;
            probindex[n0].prob = h_p[i];
            n0++;
        }
    }
    qsort(probindex, n0, sizeof(ProbIndex), compare);
    float cumulative_prob = 0.0f;
    int last_idx = n0 - 1;
    for (int i = 0; i < n0; i++) {
        cumulative_prob += probindex[i].prob;
        if (cumulative_prob > topp) {
            last_idx = i;
            break;
        }
    }
    float r = coin * cumulative_prob;
    float cdf = 0.0f;
    for (int i = 0; i <= last_idx; i++) {
        cdf += probindex[i].prob;
        if (r < cdf) {
            free(h_p);
            return probindex[i].index;
        }
    }
    free(h_p);
    return probindex[last_idx].index;
}

// GPU random number generator (simple xorshift)
static unsigned int random_u32_gpu(unsigned long long *state) {
    *state ^= *state >> 12;
    *state ^= *state << 25;
    *state ^= *state >> 27;
    return (*state * 0x2545F4914F6CDD1Dull) >> 32;
}
static float random_f32_gpu(unsigned long long *state) {
    return (random_u32_gpu(state) >> 8) / 16777216.0f;
}

// Main GPU sample function
static int sample_gpu(Sampler *sampler, float *logits) {
    int next;
    if (sampler->temperature == 0.0f) {
        next = sample_argmax_gpu(logits, sampler->vocab_size);
    } else {
        // apply temperature
        // (do this on device for efficiency)
        // Launch a simple kernel to scale logits
        int n = sampler->vocab_size;
        k_scale_logits<<<(n+255)/256, 256>>>(logits, sampler->temperature, n);
        HIP_CHECK(hipDeviceSynchronize());

        softmax_gpu(logits, sampler->vocab_size);

        float coin = random_f32_gpu(&sampler->rng_state);
        if (sampler->topp <= 0 || sampler->topp >= 1) {
            next = sample_mult_gpu(logits, sampler->vocab_size, coin);
        } else {
            next = sample_topp_gpu(logits, sampler->vocab_size, sampler->topp, sampler->probindex, coin);
        }
    }
    return next;
}