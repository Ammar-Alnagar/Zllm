// =============================================================================
// Optimized Implementation
// =============================================================================

/**
 * rmsnorm_optimized_kernel - Optimized RMSNorm with:
 *
 * 1. Single pass through input (register caching)
 * 2. Vectorized loads (float4 = 128-bit)
 * 3. Better thread utilization
 *
 * Strategy:
 * - Load input into registers while computing x²
 * - Reduce sum of squares
 * - Use cached values for normalization (no reload!)
 */
#include <cstdio>
#include <cuda_fp16.h>
#include "../include/common.cuh"

template <int BLOCK_SIZE, int ELEMENTS_PER_THREAD>
__global__ void rmsnorm_optimized_kernel(float *__restrict__ output,
                                         const float *__restrict__ input,
                                         const float *__restrict__ weight,
                                         int hidden_dim, float epsilon) {
  // One block per row
  const int row = blockIdx.x;
  const int tid = threadIdx.x;
  constexpr int WARPS_PER_BLOCK = 8; // For 256 threads
  __shared__ float warp_sums[WARPS_PER_BLOCK];

  int lane = tid % WARP_SIZE;
  int warp_id = tid / WARP_SIZE;

  // Pointers to this row
  const float *row_input = input + row * hidden_dim;
  float *row_output = output + row * hidden_dim;

  // =========================================================================
  // Step 1: Load input into registers AND compute partial x²
  // =========================================================================

  // Register array to cache input values
  float local_values[ELEMENTS_PER_THREAD];
  float sum_sq = 0.0f;

  // Vectorized load using float4 (4 floats = 16 bytes at once)
  // This achieves 128-bit memory transactions

#pragma unroll
  for (int i = 0; i < ELEMENTS_PER_THREAD; i += 4) {
    int global_idx = tid * ELEMENTS_PER_THREAD + i;

    if (global_idx + 3 < hidden_dim) {
      // Load 4 floats at once
      float4 vec = *reinterpret_cast<const float4 *>(&row_input[global_idx]);

      local_values[i + 0] = vec.x;
      local_values[i + 1] = vec.y;
      local_values[i + 2] = vec.z;
      local_values[i + 3] = vec.w;

      sum_sq += vec.x * vec.x + vec.y * vec.y + vec.z * vec.z + vec.w * vec.w;
    } else {
      // Handle boundary (when hidden_dim not divisible by 4)
      for (int j = 0; j < 4 && global_idx + j < hidden_dim; j++) {
        local_values[i + j] = row_input[global_idx + j];
        sum_sq += local_values[i + j] * local_values[i + j];
      }
    }
  }

  // =========================================================================
  // Step 2: Block reduction for sum of squares
  // =========================================================================

  constexpr int NUM_WARPS = BLOCK_SIZE / WARP_SIZE;

// Warp reduction
#pragma unroll
  for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
    sum_sq += __shfl_down_sync(0xffffffff, sum_sq, offset);
  }

  if (lane == 0) {
    warp_sums[warp_id] = sum_sq;
  }
  __syncthreads();

  // Final reduction and compute rrms
  float rrms;
  if (warp_id == 0) {
    float val = (lane < NUM_WARPS) ? warp_sums[lane] : 0.0f;

#pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
      val += __shfl_down_sync(0xffffffff, val, offset);
    }

    if (lane == 0) {
      warp_sums[0] = rsqrtf(val / (float)hidden_dim + epsilon);
    }
  }
  __syncthreads();
  rrms = warp_sums[0];

  // =========================================================================
  // Step 3: Apply normalization using cached values (NO RELOAD!)
  // =========================================================================

#pragma unroll
  for (int i = 0; i < ELEMENTS_PER_THREAD; i += 4) {
    int global_idx = tid * ELEMENTS_PER_THREAD + i;

    if (global_idx + 3 < hidden_dim) {
      // Load weights (vectorized)
      float4 w = *reinterpret_cast<const float4 *>(&weight[global_idx]);

      // Compute normalized output
      float4 out;
      out.x = local_values[i + 0] * rrms * w.x;
      out.y = local_values[i + 1] * rrms * w.y;
      out.z = local_values[i + 2] * rrms * w.z;
      out.w = local_values[i + 3] * rrms * w.w;

      // Store (vectorized)
      *reinterpret_cast<float4 *>(&row_output[global_idx]) = out;
    } else {
      // Boundary handling
      for (int j = 0; j < 4 && global_idx + j < hidden_dim; j++) {
        row_output[global_idx + j] =
            local_values[i + j] * rrms * weight[global_idx + j];
      }
    }
  }
}

// Optimized wrapper with automatic template selection
void rmsnorm_forward_optimized(float *output, const float *input,
                               const float *weight, int num_tokens,
                               int hidden_dim, float epsilon,
                               cudaStream_t stream) {
  // Choose configuration based on hidden_dim
  // Goal: Each thread handles multiple elements, using all registers

  if (hidden_dim <= 1024) {
    // Small hidden dim: 256 threads, 4 elements each
    constexpr int BLOCK_SIZE = 256;
    constexpr int ELEMENTS_PER_THREAD = 4;

    dim3 grid(num_tokens);
    dim3 block(BLOCK_SIZE);

    rmsnorm_optimized_kernel<BLOCK_SIZE, ELEMENTS_PER_THREAD>
        <<<grid, block, 0, stream>>>(output, input, weight, hidden_dim,
                                     epsilon);
  } else if (hidden_dim <= 4096) {
    // Medium hidden dim: 256 threads, 16 elements each (covers 4096)
    constexpr int BLOCK_SIZE = 256;
    constexpr int ELEMENTS_PER_THREAD = 16;

    dim3 grid(num_tokens);
    dim3 block(BLOCK_SIZE);

    rmsnorm_optimized_kernel<BLOCK_SIZE, ELEMENTS_PER_THREAD>
        <<<grid, block, 0, stream>>>(output, input, weight, hidden_dim,
                                     epsilon);
  } else {
    // Large hidden dim: 512 threads, 16 elements each (covers 8192)
    constexpr int BLOCK_SIZE = 512;
    constexpr int ELEMENTS_PER_THREAD = 16;

    dim3 grid(num_tokens);
    dim3 block(BLOCK_SIZE);

    rmsnorm_optimized_kernel<BLOCK_SIZE, ELEMENTS_PER_THREAD>
        <<<grid, block, 0, stream>>>(output, input, weight, hidden_dim,
                                     epsilon);
  }

  CUDA_CHECK_LAST();
}

// =============================================================================
// Fused RMSNorm + Residual Addition
// =============================================================================

/**
 * Common pattern in transformers:
 *   output = RMSNorm(input + residual)
 *
 * Fusing saves one memory round-trip!
 */
template <int BLOCK_SIZE, int ELEMENTS_PER_THREAD>
__global__ void rmsnorm_residual_kernel(
    float *__restrict__ output,         // [num_tokens, hidden_dim]
    float *__restrict__ residual_out,   // [num_tokens, hidden_dim] - updated
                                        // residual
    const float *__restrict__ input,    // [num_tokens, hidden_dim]
    const float *__restrict__ residual, // [num_tokens, hidden_dim]
    const float *__restrict__ weight,   // [hidden_dim]
    int hidden_dim, float epsilon) {
  const int row = blockIdx.x;
  const int tid = threadIdx.x;
  constexpr int WARPS_PER_BLOCK = 8; // For 256 threads
  __shared__ float warp_sums[WARPS_PER_BLOCK];

  int lane = tid % WARP_SIZE;
  int warp_id = tid / WARP_SIZE;

  const float *row_input = input + row * hidden_dim;
  const float *row_residual = residual + row * hidden_dim;
  float *row_output = output + row * hidden_dim;
  float *row_residual_out = residual_out + row * hidden_dim;

  // Cache for input + residual
  float local_values[ELEMENTS_PER_THREAD];
  float sum_sq = 0.0f;

// Load and add residual
#pragma unroll
  for (int i = 0; i < ELEMENTS_PER_THREAD; i += 4) {
    int idx = tid * ELEMENTS_PER_THREAD + i;

    if (idx + 3 < hidden_dim) {
      float4 in = *reinterpret_cast<const float4 *>(&row_input[idx]);
      float4 res = *reinterpret_cast<const float4 *>(&row_residual[idx]);

      // Add residual
      local_values[i + 0] = in.x + res.x;
      local_values[i + 1] = in.y + res.y;
      local_values[i + 2] = in.z + res.z;
      local_values[i + 3] = in.w + res.w;

      // Compute sum of squares
      sum_sq += local_values[i + 0] * local_values[i + 0] +
                local_values[i + 1] * local_values[i + 1] +
                local_values[i + 2] * local_values[i + 2] +
                local_values[i + 3] * local_values[i + 3];
    }
  }

  // Reduction (same as before)
  constexpr int NUM_WARPS = BLOCK_SIZE / WARP_SIZE;

#pragma unroll
  for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
    sum_sq += __shfl_down_sync(0xffffffff, sum_sq, offset);
  }

  if (lane == 0)
    warp_sums[warp_id] = sum_sq;
  __syncthreads();

  float rrms;
  if (warp_id == 0) {
    float val = (lane < NUM_WARPS) ? warp_sums[lane] : 0.0f;
#pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
      val += __shfl_down_sync(0xffffffff, val, offset);
    }
    if (lane == 0) {
      warp_sums[0] = rsqrtf(val / (float)hidden_dim + epsilon);
    }
  }
  __syncthreads();
  rrms = warp_sums[0];

// Write normalized output AND updated residual
#pragma unroll
  for (int i = 0; i < ELEMENTS_PER_THREAD; i += 4) {
    int idx = tid * ELEMENTS_PER_THREAD + i;

    if (idx + 3 < hidden_dim) {
      float4 w = *reinterpret_cast<const float4 *>(&weight[idx]);

      // Normalized output
      float4 out;
      out.x = local_values[i + 0] * rrms * w.x;
      out.y = local_values[i + 1] * rrms * w.y;
      out.z = local_values[i + 2] * rrms * w.z;
      out.w = local_values[i + 3] * rrms * w.w;
      *reinterpret_cast<float4 *>(&row_output[idx]) = out;

      // Updated residual (pre-normalized values)
      float4 res_out;
      res_out.x = local_values[i + 0];
      res_out.y = local_values[i + 1];
      res_out.z = local_values[i + 2];
      res_out.w = local_values[i + 3];
      *reinterpret_cast<float4 *>(&row_residual_out[idx]) = res_out;
    }
  }
}

// =============================================================================
// FP16 (Half Precision) Implementation
// =============================================================================

/**
 * FP16 RMSNorm - Input/output are half precision, computation in FP32
 *
 * Key points:
 * - Load as half, convert to float for computation
 * - Accumulate sum of squares in float (avoid overflow)
 * - Convert back to half for output
 */
template <int BLOCK_SIZE, int ELEMENTS_PER_THREAD>
__global__ void rmsnorm_fp16_kernel(half *__restrict__ output,
                                    const half *__restrict__ input,
                                    const half *__restrict__ weight,
                                    int hidden_dim, float epsilon) {
  const int row = blockIdx.x;
  const int tid = threadIdx.x;

  const half *row_input = input + row * hidden_dim;
  half *row_output = output + row * hidden_dim;

  // Use float for intermediate computation
  float local_values[ELEMENTS_PER_THREAD];
  float sum_sq = 0.0f;

// Load FP16, convert to FP32
#pragma unroll
  for (int i = 0; i < ELEMENTS_PER_THREAD; i += 4) {
    int idx = tid * ELEMENTS_PER_THREAD + i;

    if (idx + 3 < hidden_dim) {
      // Load 4 half values (8 bytes)
      // half2 loads are most efficient on modern GPUs
      half2 h01 = *reinterpret_cast<const half2 *>(&row_input[idx]);
      half2 h23 = *reinterpret_cast<const half2 *>(&row_input[idx + 2]);

      // Convert to float
      float2 f01 = __half22float2(h01);
      float2 f23 = __half22float2(h23);

      local_values[i + 0] = f01.x;
      local_values[i + 1] = f01.y;
      local_values[i + 2] = f23.x;
      local_values[i + 3] = f23.y;

      // Accumulate in FP32
      sum_sq += f01.x * f01.x + f01.y * f01.y + f23.x * f23.x + f23.y * f23.y;
    }
  }

  // Reduction
  constexpr int NUM_WARPS = BLOCK_SIZE / WARP_SIZE;
  __shared__ float warp_sums[NUM_WARPS];

  int lane = tid % WARP_SIZE;
  int warp_id = tid / WARP_SIZE;

#pragma unroll
  for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
    sum_sq += __shfl_down_sync(0xffffffff, sum_sq, offset);
  }

  if (lane == 0)
    warp_sums[warp_id] = sum_sq;
  __syncthreads();

  float rrms;
  if (warp_id == 0) {
    float val = (lane < NUM_WARPS) ? warp_sums[lane] : 0.0f;
#pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
      val += __shfl_down_sync(0xffffffff, val, offset);
    }
    if (lane == 0) {
      warp_sums[0] = rsqrtf(val / (float)hidden_dim + epsilon);
    }
  }
  __syncthreads();
  rrms = warp_sums[0];

// Apply normalization and convert back to FP16
#pragma unroll
  for (int i = 0; i < ELEMENTS_PER_THREAD; i += 4) {
    int idx = tid * ELEMENTS_PER_THREAD + i;

    if (idx + 3 < hidden_dim) {
      // Load weights
      half2 w01 = *reinterpret_cast<const half2 *>(&weight[idx]);
      half2 w23 = *reinterpret_cast<const half2 *>(&weight[idx + 2]);
      float2 wf01 = __half22float2(w01);
      float2 wf23 = __half22float2(w23);

      // Compute in FP32
      float2 out01, out23;
      out01.x = local_values[i + 0] * rrms * wf01.x;
      out01.y = local_values[i + 1] * rrms * wf01.y;
      out23.x = local_values[i + 2] * rrms * wf23.x;
      out23.y = local_values[i + 3] * rrms * wf23.y;

      // Convert to FP16 and store
      half2 h_out01 = __float22half2_rn(out01);
      half2 h_out23 = __float22half2_rn(out23);

      *reinterpret_cast<half2 *>(&row_output[idx]) = h_out01;
      *reinterpret_cast<half2 *>(&row_output[idx + 2]) = h_out23;
    }
  }
}

// FP16 wrapper
void rmsnorm_forward_fp16(half *output, const half *input, const half *weight,
                          int num_tokens, int hidden_dim, float epsilon,
                          cudaStream_t stream) {
  constexpr int BLOCK_SIZE = 256;
  constexpr int ELEMENTS_PER_THREAD = 16;

  dim3 grid(num_tokens);
  dim3 block(BLOCK_SIZE);

  rmsnorm_fp16_kernel<BLOCK_SIZE, ELEMENTS_PER_THREAD>
      <<<grid, block, 0, stream>>>(output, input, weight, hidden_dim, epsilon);

  CUDA_CHECK_LAST();
}
