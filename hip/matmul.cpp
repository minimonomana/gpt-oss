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

// ===== Helpers =====
#ifndef WARP_SIZE
#define WARP_SIZE warpSize        // 64 on AMD, 32 on NVIDIA
#endif

struct __align__(16) Float4 { float x,y,z,w; };

__device__ __forceinline__ int dmin(int a, int b) { return a < b ? a : b; }
__device__ __forceinline__ int dceil_div(int n, int d) { return (n + d - 1) / d; }
__device__ __forceinline__ unsigned lane_id() { return threadIdx.x & (WARP_SIZE - 1); }
__device__ __forceinline__ unsigned warp_id() { return threadIdx.x / WARP_SIZE; }

// Warp-wide reduce sum (works for both 32/64 lanes)
__device__ __forceinline__ float warp_reduce_sum(float v) {
  for (int off = WARP_SIZE >> 1; off > 0; off >>= 1) {
    v += __shfl_down(v, off, WARP_SIZE);
  }
  return v;
}

// ===== Kernel 1: Wave-per-row GEMV (shared-x, warp-reduce, optional float4) =====
template<int TILE_K=1024, bool VEC4=true>
__global__ void k_gemv_wave_row_shared(const float* __restrict__ W,  // [M,K], row-major
                                       const float* __restrict__ x,  // [K]
                                       float* __restrict__ y,        // [M]
                                       int K, int M) {
  extern __shared__ float x_s[];              // double buffer: x_s[0..TILE_K-1], x_s[TILE_K..2*TILE_K-1]
  float* x_buf0 = x_s;
  float* x_buf1 = x_s + TILE_K;

  const unsigned wid   = warp_id();           // warp index inside block
  const unsigned lane  = lane_id();           // lane index inside warp
  const unsigned waves_per_block = blockDim.x / WARP_SIZE;

  const int row = blockIdx.x * waves_per_block + wid;   // global output row
  const bool active_row = (row < M);

  // Alignment checks
  const bool smem_aligned16 = (((uintptr_t)x_s & 0xF) == 0);

  // --- Preload first tile (k0 = 0) into x_buf0 ---
  int k0 = 0;
  {
    const int tile_len = dmin(TILE_K, K - k0);
    // All threads in block cooperatively load x
    // Use blockDim.x threads to fill [0..tile_len)
    if (VEC4 && smem_aligned16 && (((uintptr_t)(x + k0) & 0xF) == 0) && tile_len >= 4) {
      int vec_elems = tile_len >> 2;
      const Float4* __restrict__ src4 = reinterpret_cast<const Float4*>(x + k0);
      Float4* __restrict__ dst4 = reinterpret_cast<Float4*>(x_buf0);
      for (int i = threadIdx.x; i < vec_elems; i += blockDim.x) dst4[i] = src4[i];
      for (int i = (vec_elems<<2) + threadIdx.x; i < tile_len; i += blockDim.x) x_buf0[i] = x[k0 + i];
    } else {
      for (int i = threadIdx.x; i < tile_len; i += blockDim.x) x_buf0[i] = x[k0 + i];
    }
  }
  __syncthreads();

  float acc = 0.0f;
  int buf = 0;

  // --- Main K loop with ping-pong buffers ---
  for (; k0 < K; k0 += TILE_K, buf ^= 1) {
    const int tile_len = dmin(TILE_K, K - k0);
    float* x_cur = (buf == 0 ? x_buf0 : x_buf1);
    float* x_nxt = (buf == 0 ? x_buf1 : x_buf0);

    // Compute on current tile
    if (active_row) {
      const float* __restrict__ wrow = W + (size_t)row * K + k0;

      // Vector path: each lane processes 4 floats per step (stride 4*WARP_SIZE)
      if (VEC4 && smem_aligned16 && (((uintptr_t)wrow & 0xF) == 0) && (((uintptr_t)x_cur & 0xF) == 0) && tile_len >= 4) {
        int vec_end = (tile_len >> 2) << 2; // largest multiple of 4
        // Loop over chunks spaced by 4*WARP_SIZE to keep alignment
        for (int jj = lane*4; jj < vec_end; jj += 4*WARP_SIZE) {
          const Float4* __restrict__ w4 = reinterpret_cast<const Float4*>(wrow + jj);
          const Float4* __restrict__ x4 = reinterpret_cast<const Float4*>(x_cur + jj);
          Float4 a = *w4;
          Float4 b = *x4;
          acc = fmaf(a.x, b.x, acc);
          acc = fmaf(a.y, b.y, acc);
          acc = fmaf(a.z, b.z, acc);
          acc = fmaf(a.w, b.w, acc);
        }
        // Tail in this tile (still lane-strided but scalar)
        for (int jj = vec_end + lane; jj < tile_len; jj += WARP_SIZE) {
          acc = fmaf(wrow[jj], x_cur[jj], acc);
        }
      } else {
        // Scalar lane-strided
        #pragma unroll 4
        for (int jj = lane; jj < tile_len; jj += WARP_SIZE) {
          acc = fmaf(wrow[jj], x_cur[jj], acc);
        }
      }
    }

    // Preload next tile (if any) into x_nxt (all threads), while other warps still computing current tile.
    int next_k0 = k0 + TILE_K;
    if (next_k0 < K) {
      const int next_len = dmin(TILE_K, K - next_k0);
      if (VEC4 && smem_aligned16 && (((uintptr_t)(x + next_k0) & 0xF) == 0) && next_len >= 4) {
        int vec_elems = next_len >> 2;
        const Float4* __restrict__ src4 = reinterpret_cast<const Float4*>(x + next_k0);
        Float4* __restrict__ dst4 = reinterpret_cast<Float4*>(x_nxt);
        for (int i = threadIdx.x; i < vec_elems; i += blockDim.x) dst4[i] = src4[i];
        for (int i = (vec_elems<<2) + threadIdx.x; i < next_len; i += blockDim.x) x_nxt[i] = x[next_k0 + i];
      } else {
        for (int i = threadIdx.x; i < next_len; i += blockDim.x) x_nxt[i] = x[next_k0 + i];
      }
    }

    __syncthreads(); // switch buffers safely
  }

  // Warp reduce & write result
  float sum = warp_reduce_sum(acc);
  if (lane == 0 && active_row) y[row] = sum;
}

// ===== Kernel 2: Split-K version (atomicAdd on Y) =====
template<int TILE_K=1024, bool VEC4=true>
__global__ void k_gemv_wave_row_shared_splitK(const float* __restrict__ W,
                                              const float* __restrict__ x,
                                              float* __restrict__ y,   // output (atomicAdd)
                                              int K, int M,
                                              int splitK) {
  extern __shared__ float x_s[];
  float* x_buf0 = x_s;
  float* x_buf1 = x_s + TILE_K;

  const unsigned wid   = warp_id();
  const unsigned lane  = lane_id();
  const unsigned waves_per_block = blockDim.x / WARP_SIZE;

  const int row = blockIdx.x * waves_per_block + wid;
  const bool active_row = (row < M);

  // This block's K-partition
  const int part   = blockIdx.y;           // 0..splitK-1
  const int span   = dceil_div(K, splitK);
  const int k_begin = part * span;
  const int k_end   = dmin(K, k_begin + span);

  if (k_begin >= k_end) return;            // empty slice (still all threads in block reach here together)

  const bool smem_aligned16 = (((uintptr_t)x_s & 0xF) == 0);

  // Preload first tile
  int k0 = k_begin;
  {
    const int tile_len = dmin(TILE_K, k_end - k0);
    if (VEC4 && smem_aligned16 && (((uintptr_t)(x + k0) & 0xF) == 0) && tile_len >= 4) {
      int vec_elems = tile_len >> 2;
      const Float4* __restrict__ src4 = reinterpret_cast<const Float4*>(x + k0);
      Float4* __restrict__ dst4 = reinterpret_cast<Float4*>(x_buf0);
      for (int i = threadIdx.x; i < vec_elems; i += blockDim.x) dst4[i] = src4[i];
      for (int i = (vec_elems<<2) + threadIdx.x; i < tile_len; i += blockDim.x) x_buf0[i] = x[k0 + i];
    } else {
      for (int i = threadIdx.x; i < tile_len; i += blockDim.x) x_buf0[i] = x[k0 + i];
    }
  }
  __syncthreads();

  float acc = 0.0f;
  int buf = 0;

  for (; k0 < k_end; k0 += TILE_K, buf ^= 1) {
    const int tile_len = dmin(TILE_K, k_end - k0);
    float* x_cur = (buf == 0 ? x_buf0 : x_buf1);
    float* x_nxt = (buf == 0 ? x_buf1 : x_buf0);

    if (active_row) {
      const float* __restrict__ wrow = W + (size_t)row * K + k0;

      if (VEC4 && smem_aligned16 && (((uintptr_t)wrow & 0xF) == 0) && (((uintptr_t)x_cur & 0xF) == 0) && tile_len >= 4) {
        int vec_end = (tile_len >> 2) << 2;
        for (int jj = lane*4; jj < vec_end; jj += 4*WARP_SIZE) {
          const Float4* __restrict__ w4 = reinterpret_cast<const Float4*>(wrow + jj);
          const Float4* __restrict__ x4 = reinterpret_cast<const Float4*>(x_cur + jj);
          Float4 a = *w4;
          Float4 b = *x4;
          acc = fmaf(a.x, b.x, acc);
          acc = fmaf(a.y, b.y, acc);
          acc = fmaf(a.z, b.z, acc);
          acc = fmaf(a.w, b.w, acc);
        }
        for (int jj = vec_end + lane; jj < tile_len; jj += WARP_SIZE) {
          acc = fmaf(wrow[jj], x_cur[jj], acc);
        }
      } else {
        #pragma unroll 4
        for (int jj = lane; jj < tile_len; jj += WARP_SIZE) {
          acc = fmaf(wrow[jj], x_cur[jj], acc);
        }
      }
    }

    // Preload next tile (if any)
    int next_k0 = k0 + TILE_K;
    if (next_k0 < k_end) {
      const int next_len = dmin(TILE_K, k_end - next_k0);
      if (VEC4 && smem_aligned16 && (((uintptr_t)(x + next_k0) & 0xF) == 0) && next_len >= 4) {
        int vec_elems = next_len >> 2;
        const Float4* __restrict__ src4 = reinterpret_cast<const Float4*>(x + next_k0);
        Float4* __restrict__ dst4 = reinterpret_cast<Float4*>(x_nxt);
        for (int i = threadIdx.x; i < vec_elems; i += blockDim.x) dst4[i] = src4[i];
        for (int i = (vec_elems<<2) + threadIdx.x; i < next_len; i += blockDim.x) x_nxt[i] = x[next_k0 + i];
      } else {
        for (int i = threadIdx.x; i < next_len; i += blockDim.x) x_nxt[i] = x[next_k0 + i];
      }
    }

    __syncthreads();
  }

  float sum = warp_reduce_sum(acc);
  if (lane == 0 && active_row) {
    // Accumulate this K-slice into y[row]
    atomicAdd(&y[row], sum);
  }
}

// ===== Launchers =====

// Wave-per-row launcher
// waves_per_block: 2..8 thường tốt (tuỳ arch); TILE_K: 1024/2048/4096
template<int TILE_K=1024, bool VEC4=true>
void gemv_wave_row(float* y, const float* x, const float* W, int K, int M,
                   int waves_per_block = 4, hipStream_t stream = 0) {
  const int threads = waves_per_block * WARP_SIZE;
  dim3 block(threads, 1, 1);
  dim3 grid((M + waves_per_block - 1) / waves_per_block, 1, 1);
  // double buffer in LDS
  size_t shmem = 2 * TILE_K * sizeof(float);

  // Zero init y (caller may do this too)
  hipMemsetAsync(y, 0, M * sizeof(float), stream);

  hipLaunchKernelGGL((k_gemv_wave_row_shared<TILE_K, VEC4>),
                     grid, block, shmem, stream,
                     W, x, y, K, M);
}

// Split-K launcher (atomicAdd). Choose splitK=2..8 khi K rất lớn để tăng occupancy.
template<int TILE_K=1024, bool VEC4=true>
void gemv_splitK(float* y, const float* x, const float* W, int K, int M,
                 int splitK, int waves_per_block = 4, hipStream_t stream = 0) {
  const int threads = waves_per_block * WARP_SIZE;
  dim3 block(threads, 1, 1);
  dim3 grid((M + waves_per_block - 1) / waves_per_block, splitK, 1);
  size_t shmem = 2 * TILE_K * sizeof(float);

  hipMemsetAsync(y, 0, M * sizeof(float), stream);

  hipLaunchKernelGGL((k_gemv_wave_row_shared_splitK<TILE_K, VEC4>),
                     grid, block, shmem, stream,
                     W, x, y, K, M, splitK);
}

static void gemv_gpu(float *y,              // out: [out_features]
                     const float *x,        // [in_features]
                     const float *W,        // [out_features, in_features] row-major
                     int in_features,       // K
                     int out_features) {    // M
  const int K = in_features;
  const int M = out_features;

  // Heuristic nhẹ:
  // - TILE_K=2048 (8 KiB), dùng double-buffer => 16 KiB LDS/block
  // - wavefront AMD = 64 lanes; 4 waves/block = 256 threads
  constexpr int TILE_K = 2048;
  const int waves_per_block = 4;
  const int WARP_AMD = 64;                   // wavefront size trên AMD
  const dim3 block(waves_per_block * WARP_AMD, 1, 1);
  size_t shmem_bytes = 2ULL * TILE_K * sizeof(float);

  // splitK khi K rất lớn để tăng occupancy (atomicAdd gộp kết quả)
  int splitK = 1;
  if      (K >= 65536) splitK = 8;
  else if (K >= 32768) splitK = 4;
  else if (K >= 16384) splitK = 2;

  if (splitK == 1) {
    // Wave-per-row, không cần zero y (kernel ghi đè)
    const dim3 grid((M + waves_per_block - 1) / waves_per_block, 1, 1);
    hipLaunchKernelGGL((k_gemv_wave_row_shared<TILE_K, /*VEC4=*/true>),
                       grid, block, shmem_bytes, 0,
                       W, x, y, K, M);
  } else {
    // Split-K: cần zero y trước vì kernel dùng atomicAdd
    hipMemsetAsync(y, 0, (size_t)M * sizeof(float), 0);
    const dim3 grid((M + waves_per_block - 1) / waves_per_block, splitK, 1);
    hipLaunchKernelGGL((k_gemv_wave_row_shared_splitK<TILE_K, /*VEC4=*/true>),
                       grid, block, shmem_bytes, 0,
                       W, x, y, K, M, splitK);
  }
}

// ---------- Mixed-precision GEMV: x (FP32) * W (BF16, packed uint16_t on device) -> y (FP32) ----------

// Wave-per-row kernel but W is BF16 packed as uint16_t
template<int TILE_K=1024, bool VEC4=true>
__global__ void k_gemv_wave_row_shared_bf16(__half* __restrict__ W, // [M,K] packed BF16
                                            const float* __restrict__ x,   // [K] FP32
                                            float* __restrict__ y,         // [M] FP32
                                            int K, int M) {
  extern __shared__ float x_s[]; // double buffer
  float* x_buf0 = x_s;
  float* x_buf1 = x_s + TILE_K;

  const unsigned wid   = warp_id();
  const unsigned lane  = lane_id();
  const unsigned waves_per_block = blockDim.x / WARP_SIZE;

  const int row = blockIdx.x * waves_per_block + wid;
  const bool active_row = (row < M);

  const bool smem_aligned16 = (((uintptr_t)x_s & 0xF) == 0);

  int k0 = 0;
  {
    const int tile_len = dmin(TILE_K, K - k0);
    if (VEC4 && smem_aligned16 && (((uintptr_t)(x + k0) & 0xF) == 0) && tile_len >= 4) {
      int vec_elems = tile_len >> 2;
      const Float4* __restrict__ src4 = reinterpret_cast<const Float4*>(x + k0);
      Float4* __restrict__ dst4 = reinterpret_cast<Float4*>(x_buf0);
      for (int i = threadIdx.x; i < vec_elems; i += blockDim.x) dst4[i] = src4[i];
      for (int i = (vec_elems<<2) + threadIdx.x; i < tile_len; i += blockDim.x) x_buf0[i] = x[k0 + i];
    } else {
      for (int i = threadIdx.x; i < tile_len; i += blockDim.x) x_buf0[i] = x[k0 + i];
    }
  }
  __syncthreads();

  float acc = 0.0f;
  int buf = 0;

  for (; k0 < K; k0 += TILE_K, buf ^= 1) {
    const int tile_len = dmin(TILE_K, K - k0);
    float* x_cur = (buf == 0 ? x_buf0 : x_buf1);
    float* x_nxt = (buf == 0 ? x_buf1 : x_buf0);

    if (active_row) {
      // W is uint16_t packed BF16; row pointer:
      __half* __restrict__ wrow = W + (size_t)row * (size_t)K + k0;

      if (VEC4 && smem_aligned16 && (((uintptr_t)wrow & 0xF) == 0) && (((uintptr_t)x_cur & 0xF) == 0) && tile_len >= 4) {
        int vec_end = (tile_len >> 2) << 2;
        // process 4-elements at a time
        for (int jj = lane*4; jj < vec_end; jj += WARP_SIZE*4) {
          // manual unroll: convert bf16->float, then fmaf
          float a0 = (float)(wrow[jj+0]);
          float a1 = (float)(wrow[jj+1]);
          float a2 = (float)(wrow[jj+2]);
          float a3 = (float)(wrow[jj+3]);
          const Float4* __restrict__ x4 = reinterpret_cast<const Float4*>(x_cur + jj);
          Float4 b = *x4;
          Float4 a = {a0, a1, a2, a3};
          acc = fmaf(a.x, b.x, acc);
          acc = fmaf(a.y, b.y, acc);
          acc = fmaf(a.z, b.z, acc);
          acc = fmaf(a.w, b.w, acc);
          // acc = fmaf(a0, x_cur[jj+0], acc);
          // acc = fmaf(a1, x_cur[jj+1], acc);
          // acc = fmaf(a2, x_cur[jj+2], acc);
          // acc = fmaf(a3, x_cur[jj+3], acc);
        }
        for (int jj = ((vec_end) + lane); jj < tile_len; jj += WARP_SIZE) {
          float a = (float)(wrow[jj]);
          acc = fmaf(a, x_cur[jj], acc);
        }
      } else {
        #pragma unroll 4
        for (int jj = lane; jj < tile_len; jj += WARP_SIZE) {
          float a = (float)(wrow[jj]);
          acc = fmaf(a, x_cur[jj], acc);
        }
      }
    }

    int next_k0 = k0 + TILE_K;
    if (next_k0 < K) {
      const int next_len = dmin(TILE_K, K - next_k0);
      if (VEC4 && smem_aligned16 && (((uintptr_t)(x + next_k0) & 0xF) == 0) && next_len >= 4) {
        int vec_elems = next_len >> 2;
        const Float4* __restrict__ src4 = reinterpret_cast<const Float4*>(x + next_k0);
        Float4* __restrict__ dst4 = reinterpret_cast<Float4*>(x_nxt);
        for (int i = threadIdx.x; i < vec_elems; i += blockDim.x) dst4[i] = src4[i];
        for (int i = (vec_elems<<2) + threadIdx.x; i < next_len; i += blockDim.x) x_nxt[i] = x[next_k0 + i];
      } else {
        for (int i = threadIdx.x; i < next_len; i += blockDim.x) x_nxt[i] = x[next_k0 + i];
      }
    }

    __syncthreads();
  }

  float sum = warp_reduce_sum(acc);
  if (lane == 0 && active_row) {
    // accumulate (same as normal kernel)
    atomicAdd(&y[row], sum);
    // y[row] = sum;
  }
}

// Split-K version (atomicAdd) for BF16 - similar idea
template<int TILE_K=1024, bool VEC4=true>
__global__ void k_gemv_wave_row_shared_splitK_bf16(__half* __restrict__ W,
                                                  const float* __restrict__ x,
                                                  float* __restrict__ y,
                                                  int K, int M, int splitK) {
  extern __shared__ float x_s[];
  float* x_buf0 = x_s;
  float* x_buf1 = x_s + TILE_K;

  const unsigned wid   = warp_id();
  const unsigned lane  = lane_id();
  const unsigned waves_per_block = blockDim.x / WARP_SIZE;

  const int row = blockIdx.x * waves_per_block + wid;
  const bool active_row = (row < M);

  const int part   = blockIdx.y;           // 0..splitK-1
  const int span   = dceil_div(K, splitK);
  const int k_begin = part * span;
  const int k_end   = dmin(K, k_begin + span);

  if (k_begin >= k_end) return;

  const bool smem_aligned16 = (((uintptr_t)x_s & 0xF) == 0);

  int k0 = k_begin;
  {
    const int tile_len = dmin(TILE_K, k_end - k0);
    if (VEC4 && smem_aligned16 && (((uintptr_t)(x + k0) & 0xF) == 0) && tile_len >= 4) {
      int vec_elems = tile_len >> 2;
      const Float4* __restrict__ src4 = reinterpret_cast<const Float4*>(x + k0);
      Float4* __restrict__ dst4 = reinterpret_cast<Float4*>(x_buf0);
      for (int i = threadIdx.x; i < vec_elems; i += blockDim.x) dst4[i] = src4[i];
      for (int i = (vec_elems<<2) + threadIdx.x; i < tile_len; i += blockDim.x) x_buf0[i] = x[k0 + i];
    } else {
      for (int i = threadIdx.x; i < tile_len; i += blockDim.x) x_buf0[i] = x[k0 + i];
    }
  }
  __syncthreads();

  float acc = 0.0f;
  int buf = 0;

  for (; k0 < k_end; k0 += TILE_K, buf ^= 1) {
    const int tile_len = dmin(TILE_K, k_end - k0);
    float* x_cur = (buf == 0 ? x_buf0 : x_buf1);
    float* x_nxt = (buf == 0 ? x_buf1 : x_buf0);

    if (active_row) {
      const __half* __restrict__ wrow = W + (size_t)row * (size_t)K + k0;

      if (VEC4 && smem_aligned16 && (((uintptr_t)wrow & 0xF) == 0) && (((uintptr_t)x_cur & 0xF) == 0) && tile_len >= 4) {
        int vec_end = (tile_len >> 2) << 2;
        for (int jj = lane*4; jj < vec_end; jj += WARP_SIZE*4) {
          float a0 = (float)(wrow[jj+0]);
          float a1 = (float)(wrow[jj+1]);
          float a2 = (float)(wrow[jj+2]);
          float a3 = (float)(wrow[jj+3]);
          const Float4* __restrict__ x4 = reinterpret_cast<const Float4*>(x_cur + jj);
          Float4 b = *x4;
          Float4 a = {a0, a1, a2, a3};
          acc = fmaf(a.x, b.x, acc);
          acc = fmaf(a.y, b.y, acc);
          acc = fmaf(a.z, b.z, acc);
          acc = fmaf(a.w, b.w, acc);
          // acc = fmaf(a0, x_cur[jj+0], acc);
          // acc = fmaf(a1, x_cur[jj+1], acc);
          // acc = fmaf(a2, x_cur[jj+2], acc);
          // acc = fmaf(a3, x_cur[jj+3], acc);
        }
        for (int jj = ((vec_end) + lane); jj < tile_len; jj += WARP_SIZE) {
          float a = (float)(wrow[jj]);
          acc = fmaf(a, x_cur[jj], acc);
        }
      } else {
        #pragma unroll 4
        for (int jj = lane; jj < tile_len; jj += WARP_SIZE) {
          float a = (float)(wrow[jj]);
          acc = fmaf(a, x_cur[jj], acc);
        }
      }
    }

    int next_k0 = k0 + TILE_K;
    if (next_k0 < k_end) {
      const int next_len = dmin(TILE_K, k_end - next_k0);
      if (VEC4 && smem_aligned16 && (((uintptr_t)(x + next_k0) & 0xF) == 0) && next_len >= 4) {
        int vec_elems = next_len >> 2;
        const Float4* __restrict__ src4 = reinterpret_cast<const Float4*>(x + next_k0);
        Float4* __restrict__ dst4 = reinterpret_cast<Float4*>(x_nxt);
        for (int i = threadIdx.x; i < vec_elems; i += blockDim.x) dst4[i] = src4[i];
        for (int i = (vec_elems<<2) + threadIdx.x; i < next_len; i += blockDim.x) x_nxt[i] = x[next_k0 + i];
      } else {
        for (int i = threadIdx.x; i < next_len; i += blockDim.x) x_nxt[i] = x[next_k0 + i];
      }
    }

    __syncthreads();
  }

  float sum = warp_reduce_sum(acc);
  if (lane == 0 && active_row) atomicAdd(&y[row], sum);
}

// Launcher for BF16 mixed GEMV: similar heuristics to gemv_gpu
static void gemv_fp16_weights(float *y,             // out: [M]
                                 const float *x,       // in: [K]
                                 __half *W_fp16,// weights BF16 (packed) [M,K] row-major
                                 int in_features,      // K
                                 int out_features) {   // M
  const int K = in_features;
  const int M = out_features;

  constexpr int TILE_K = 2048;
  const int waves_per_block = 4;
  const int WARP_AMD = 64;
  const dim3 block(waves_per_block * WARP_AMD, 1, 1);
  size_t shmem_bytes = 2ULL * TILE_K * sizeof(float);
  // const __half *W_bf16 = reinterpret_cast<const __half*>(W_fp16);
  hipMemsetAsync(y, 0, (size_t)M * sizeof(float), 0);

  int splitK = 1;
  if      (K >= 65536) splitK = 8;
  else if (K >= 32768) splitK = 4;
  else if (K >= 16384) splitK = 2;

  if (splitK == 1) {
    const dim3 grid((M + waves_per_block - 1) / waves_per_block, 1, 1);
    hipLaunchKernelGGL((k_gemv_wave_row_shared_bf16<TILE_K, true>),
                       grid, block, shmem_bytes, 0,
                       W_fp16, x, y, K, M);
  } else {
    // hipMemset(y, 0, (size_t)M * sizeof(float));
    const dim3 grid((M + waves_per_block - 1) / waves_per_block, splitK, 1);
    hipLaunchKernelGGL((k_gemv_wave_row_shared_splitK_bf16<TILE_K, true>),
                       grid, block, shmem_bytes, 0,
                       W_fp16, x, y, K, M, splitK);
  }
}
