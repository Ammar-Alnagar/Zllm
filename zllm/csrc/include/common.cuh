// =============================================================================
// common.cuh - Common CUDA Utilities
// =============================================================================
// This header provides common utilities used across all CUDA kernels:
// - Error checking macros
// - Warp-level primitives
// - Memory alignment helpers
// - Common constants
// =============================================================================

#pragma once

#include <cstdio>
#include <cstdlib>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

// =============================================================================
// CUDA Error Checking
// =============================================================================

/**
 * CUDA_CHECK - Macro for checking CUDA API call results
 *
 * Usage:
 *     CUDA_CHECK(cudaMalloc(&ptr, size));
 *
 * If the call fails, prints error message with file/line and exits
 */
#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t err = call;                                                    \
    if (err != cudaSuccess) {                                                  \
      fprintf(stderr, "CUDA Error at %s:%d - %s\n", __FILE__, __LINE__,        \
              cudaGetErrorString(err));                                        \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

/**
 * CUDA_CHECK_LAST - Check for errors after kernel launch
 *
 * Usage:
 *     my_kernel<<<grid, block>>>(...);
 *     CUDA_CHECK_LAST();
 */
#define CUDA_CHECK_LAST()                                                      \
  do {                                                                         \
    cudaError_t err = cudaGetLastError();                                      \
    if (err != cudaSuccess) {                                                  \
      fprintf(stderr, "CUDA Kernel Error at %s:%d - %s\n", __FILE__, __LINE__, \
              cudaGetErrorString(err));                                        \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

// =============================================================================
// Constants
// =============================================================================

// Warp size is fixed at 32 threads for all NVIDIA GPUs
constexpr int WARP_SIZE = 32;

// Maximum threads per block (hardware limit)
constexpr int MAX_THREADS_PER_BLOCK = 512;

// Maximum shared memory per block (48KB typical, 164KB on Hopper)
constexpr int MAX_SHARED_MEMORY = 48 * 512;

// Cache line size for memory coalescing (128 bytes)
constexpr int CACHE_LINE_SIZE = 128;

// KV cache block size (16 tokens per block)
constexpr int KV_BLOCK_SIZE = 16;

// =============================================================================
// Type Aliases
// =============================================================================

using fp16 = half;          // FP16 type
using bf16 = __nv_bfloat16; // BF16 type

// =============================================================================
// Device Functions - Warp Primitives
// =============================================================================

/**
 * warp_reduce_sum - Sum reduction within a warp
 *
 * Uses shuffle instructions for efficient intra-warp communication.
 * All threads in the warp must participate.
 *
 * @param val: Value to reduce from each thread
 * @return: Sum of all values in the warp (in lane 0, broadcast to all)
 */
template <typename T> __device__ __forceinline__ T warp_reduce_sum(T val) {
// Shuffle down and add, halving the distance each iteration
// This creates a tree reduction pattern
#pragma unroll
  for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
    val += __shfl_down_sync(0xffffffff, val, offset);
  }
  // Broadcast result from lane 0 to all lanes
  return __shfl_sync(0xffffffff, val, 0);
}

/**
 * warp_reduce_max - Maximum reduction within a warp
 *
 * @param val: Value to reduce from each thread
 * @return: Maximum value in the warp
 */
template <typename T> __device__ __forceinline__ T warp_reduce_max(T val) {
#pragma unroll
  for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
    val = max(val, __shfl_down_sync(0xffffffff, val, offset));
  }
  return __shfl_sync(0xffffffff, val, 0);
}

/**
 * block_reduce_sum - Sum reduction across an entire block
 *
 * Uses shared memory for inter-warp communication.
 *
 * @param val: Value to reduce from each thread
 * @param shared: Pointer to shared memory (size = num_warps)
 * @return: Sum of all values in the block (in thread 0, broadcast to all)
 */
template <typename T>
__device__ __forceinline__ T block_reduce_sum(T val, T *shared) {
  const int lane = threadIdx.x % WARP_SIZE;
  const int warp_id = threadIdx.x / WARP_SIZE;
  const int num_warps = blockDim.x / WARP_SIZE;

  // First, reduce within each warp
  val = warp_reduce_sum(val);

  // Write warp results to shared memory
  if (lane == 0) {
    shared[warp_id] = val;
  }
  __syncthreads();

  // First warp reduces all warp results
  if (warp_id == 0) {
    val = (lane < num_warps) ? shared[lane] : T(0);
    val = warp_reduce_sum(val);
  }
  __syncthreads();

  // Broadcast final result
  return __shfl_sync(0xffffffff, val, 0);
}

// =============================================================================
// Memory Helpers
// =============================================================================

/**
 * align_up - Align size up to nearest multiple of alignment
 *
 * @param size: Size to align
 * @param alignment: Alignment boundary (must be power of 2)
 * @return: Aligned size
 */
__host__ __device__ __forceinline__ size_t align_up(size_t size,
                                                    size_t alignment) {
  return (size + alignment - 1) & ~(alignment - 1);
}

/**
 * div_ceil - Integer division, rounding up
 *
 * @param a: Dividend
 * @param b: Divisor
 * @return: Ceiling of a/b
 */
template <typename T> __host__ __device__ __forceinline__ T div_ceil(T a, T b) {
  return (a + b - 1) / b;
}

// =============================================================================
// Numeric Helpers
// =============================================================================

/**
 * fast_rsqrt - Fast reciprocal square root (1/sqrt(x))
 *
 * Uses hardware rsqrt instruction for speed
 */
__device__ __forceinline__ float fast_rsqrt(float x) { return rsqrtf(x); }

/**
 * silu - Sigmoid Linear Unit activation (Swish)
 *
 * silu(x) = x * sigmoid(x) = x / (1 + exp(-x))
 */
__device__ __forceinline__ float silu(float x) { return x / (1.0f + expf(-x)); }

// =============================================================================
// FP16/BF16 Conversion Helpers
// =============================================================================

__device__ __forceinline__ float fp16_to_float(fp16 x) {
  return __half2float(x);
}

__device__ __forceinline__ fp16 float_to_fp16(float x) {
  return __float2half(x);
}

__device__ __forceinline__ float bf16_to_float(bf16 x) {
  return __bfloat162float(x);
}

__device__ __forceinline__ bf16 float_to_bf16(float x) {
  return __float2bfloat16(x);
}

// =============================================================================
// Debug Helpers
// =============================================================================

#ifdef DEBUG_KERNELS
#define KERNEL_DEBUG_PRINT(fmt, ...)                                           \
  if (threadIdx.x == 0 && blockIdx.x == 0) {                                   \
    printf("[KERNEL DEBUG] " fmt "\n", ##__VA_ARGS__);                         \
  }
#else
#define KERNEL_DEBUG_PRINT(fmt, ...)
#endif
