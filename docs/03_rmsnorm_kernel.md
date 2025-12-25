# Phase 1: RMSNorm Kernel Implementation

## Table of Contents

1. [Understanding RMSNorm](#understanding-rmsnorm)
2. [Mathematical Background](#mathematical-background)
3. [Implementation Strategy](#implementation-strategy)
4. [Naive Implementation](#naive-implementation)
5. [Optimized Implementation](#optimized-implementation)
6. [Fused RMSNorm Kernel](#fused-rmsnorm-kernel)
7. [Testing and Verification](#testing-and-verification)

---

## Understanding RMSNorm

**Root Mean Square Layer Normalization (RMSNorm)** is a simplification of LayerNorm used in Qwen, LLaMA, and other modern LLMs. It normalizes activations without mean centering.

```
                    RMSNorm vs LayerNorm

LayerNorm:
┌─────────────────────────────────────────────────────────┐
│  output = γ × (x - mean(x)) / sqrt(var(x) + ε) + β     │
│                                                         │
│  Requires: mean, variance computation + scale + shift   │
└─────────────────────────────────────────────────────────┘

RMSNorm:
┌─────────────────────────────────────────────────────────┐
│  output = γ × x / sqrt(mean(x²) + ε)                   │
│                                                         │
│  Requires: mean of squares only + scale (no shift!)    │
└─────────────────────────────────────────────────────────┘

RMSNorm is ~10-15% faster because:
- No mean subtraction needed
- No bias (β) term
- Single reduction instead of two
```

### Where RMSNorm is Used

```
Transformer Decoder Block:
┌────────────────────────────────────────┐
│                                        │
│   Input ──▶ [RMSNorm] ──▶ Attention   │  ◀── Pre-normalization
│                    │                   │
│                    └──────────────┐    │
│                                   │    │
│   ┌───────────────────────────────┘    │
│   │                                    │
│   └──▶ [RMSNorm] ──▶ FFN              │  ◀── Pre-normalization
│                                        │
└────────────────────────────────────────┘

In Qwen3:
- Pre-attention RMSNorm
- Pre-FFN RMSNorm
- Final output RMSNorm
```

---

## Mathematical Background

### RMSNorm Formula

For an input vector **x** of dimension **d**:

```
RMS(x) = sqrt(1/d × Σᵢ xᵢ²)

output = γ × x / RMS(x)

Where:
- x: Input tensor of shape [batch, seq_len, hidden_dim]
- γ: Learned scale parameter of shape [hidden_dim]
- ε: Small constant for numerical stability (e.g., 1e-6)
```

### Computation Steps

```
Step 1: Compute sum of squares
        ss = Σᵢ xᵢ²

Step 2: Compute RMS with epsilon
        rms = sqrt(ss / d + ε)

Step 3: Compute reciprocal (1/rms)
        rrms = 1 / rms = rsqrt(ss / d + ε)

Step 4: Scale each element
        outputᵢ = γᵢ × xᵢ × rrms
```

### Numerical Considerations

```
                    Numerical Stability

Problem: For FP16/BF16, sum of squares can overflow!

Example (FP16 max ≈ 65504):
- hidden_dim = 4096
- If each x² ≈ 16, then sum = 4096 × 16 = 65536 > 65504 💥

Solution 1: Accumulate in FP32
            Even if inputs are FP16, use float for reduction

Solution 2: Scale inputs
            Divide inputs by sqrt(hidden_dim) before squaring

We use Solution 1 (most common approach)
```

---

## Implementation Strategy

### Memory Access Pattern

```
Input tensor: [batch_size, seq_len, hidden_dim]

              hidden_dim (4096) ──────────────────────▶
          ┌───────────────────────────────────────────────┐
   batch  │ ▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒ │ row 0
    ×     │ ▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒ │ row 1
 seq_len  │ ▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒ │ row 2
    │     │ ...                                          │
    ▼     └───────────────────────────────────────────────┘

Strategy:
- Each block processes ONE row (one token's hidden states)
- Block size = 256 or 512 threads
- Each thread handles multiple elements (hidden_dim / block_size)
```

### Kernel Organization

```
                    RMSNorm Kernel Structure

Grid:  <<<batch_size × seq_len, 1>>>   (one block per row)
Block: <<<1, 256>>> or <<<1, 512>>>    (threads process hidden_dim)

Block 0 ──▶ Row 0: Reduce, Normalize
Block 1 ──▶ Row 1: Reduce, Normalize
Block 2 ──▶ Row 2: Reduce, Normalize
   ...
Block N ──▶ Row N: Reduce, Normalize

Each block:
1. Load elements → compute x² → reduce sum
2. Compute rsqrt(sum/d + ε)
3. Load elements again → multiply by scale and rrms → store
```

---

## Naive Implementation

Let's start with a straightforward implementation:

Create file: `mini_vllm/csrc/kernels/rmsnorm.cuh`

```cuda
// =============================================================================
// rmsnorm.cuh - RMSNorm Kernel Header
// =============================================================================

#pragma once

#include "common.cuh"
#include <cuda_runtime.h>
#include <cuda_fp16.h>

namespace mini_vllm {

/**
 * RMSNorm forward pass
 *
 * @param output: Output tensor [num_tokens, hidden_dim]
 * @param input: Input tensor [num_tokens, hidden_dim]
 * @param weight: Scale parameter γ [hidden_dim]
 * @param num_tokens: Number of tokens (batch × seq_len)
 * @param hidden_dim: Hidden dimension
 * @param epsilon: Numerical stability constant
 * @param stream: CUDA stream
 */
void rmsnorm_forward(
    float* output,
    const float* input,
    const float* weight,
    int num_tokens,
    int hidden_dim,
    float epsilon,
    cudaStream_t stream = nullptr
);

// FP16 version
void rmsnorm_forward_fp16(
    half* output,
    const half* input,
    const half* weight,
    int num_tokens,
    int hidden_dim,
    float epsilon,
    cudaStream_t stream = nullptr
);

} // namespace mini_vllm
```

Create file: `mini_vllm/csrc/kernels/rmsnorm.cu`

```cuda
// =============================================================================
// rmsnorm.cu - RMSNorm Kernel Implementation
// =============================================================================

#include "rmsnorm.cuh"

namespace mini_vllm {

// =============================================================================
// Naive Implementation
// =============================================================================

/**
 * rmsnorm_naive_kernel - Simple RMSNorm without optimization
 *
 * Each block processes one row of the input tensor.
 * Uses shared memory for block reduction.
 *
 * Performance characteristics:
 * - Simple but not optimal
 * - Reads input twice (reduction + normalization)
 * - Good for understanding the algorithm
 */
__global__ void rmsnorm_naive_kernel(
    float* __restrict__ output,      // [num_tokens, hidden_dim]
    const float* __restrict__ input, // [num_tokens, hidden_dim]
    const float* __restrict__ weight,// [hidden_dim]
    int hidden_dim,
    float epsilon
) {
    // One block per row (token)
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const int num_threads = blockDim.x;

    // Pointers to this row
    const float* row_input = input + row * hidden_dim;
    float* row_output = output + row * hidden_dim;

    // =========================================================================
    // Step 1: Compute sum of squares (reduction)
    // =========================================================================

    // Each thread computes partial sum of squares
    float thread_sum = 0.0f;

    // Stride through hidden dimension
    // If hidden_dim=4096 and num_threads=256:
    // Thread 0 handles: 0, 256, 512, 768, ..., 3840 (16 elements)
    // Thread 1 handles: 1, 257, 513, 769, ..., 3841
    for (int i = tid; i < hidden_dim; i += num_threads) {
        float val = row_input[i];
        thread_sum += val * val;
    }

    // =========================================================================
    // Step 2: Block-level reduction
    // =========================================================================

    // Shared memory for reduction (one float per warp)
    constexpr int WARPS_PER_BLOCK = 8;  // For 256 threads
    __shared__ float warp_sums[WARPS_PER_BLOCK];

    int lane = tid % WARP_SIZE;
    int warp_id = tid / WARP_SIZE;

    // Warp-level reduction using shuffle
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        thread_sum += __shfl_down_sync(0xffffffff, thread_sum, offset);
    }

    // First lane of each warp writes to shared memory
    if (lane == 0) {
        warp_sums[warp_id] = thread_sum;
    }
    __syncthreads();

    // First warp reduces warp sums
    float total_sum = 0.0f;
    if (warp_id == 0) {
        float val = (lane < WARPS_PER_BLOCK) ? warp_sums[lane] : 0.0f;

        #pragma unroll
        for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
            val += __shfl_down_sync(0xffffffff, val, offset);
        }

        if (lane == 0) {
            // Compute: 1 / sqrt(mean(x²) + epsilon)
            total_sum = rsqrtf(val / (float)hidden_dim + epsilon);
            warp_sums[0] = total_sum;  // Store for all threads
        }
    }
    __syncthreads();

    // All threads read the final result
    float rrms = warp_sums[0];

    // =========================================================================
    // Step 3: Normalize and apply scale
    // =========================================================================

    for (int i = tid; i < hidden_dim; i += num_threads) {
        float val = row_input[i];
        float w = weight[i];
        row_output[i] = val * rrms * w;
    }
}

// Wrapper function
void rmsnorm_forward(
    float* output,
    const float* input,
    const float* weight,
    int num_tokens,
    int hidden_dim,
    float epsilon,
    cudaStream_t stream
) {
    // Configuration
    const int BLOCK_SIZE = 256;

    dim3 grid(num_tokens);
    dim3 block(BLOCK_SIZE);

    rmsnorm_naive_kernel<<<grid, block, 0, stream>>>(
        output, input, weight, hidden_dim, epsilon
    );

    CUDA_CHECK_LAST();
}

} // namespace mini_vllm
```

---

## Optimized Implementation

The naive kernel has several issues:

1. Reads input twice (once for reduction, once for normalization)
2. Not fully utilizing memory bandwidth
3. No vectorized loads

Let's optimize:

Add to `rmsnorm.cu`:

```cuda
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
template<int BLOCK_SIZE, int ELEMENTS_PER_THREAD>
__global__ void rmsnorm_optimized_kernel(
    float* __restrict__ output,
    const float* __restrict__ input,
    const float* __restrict__ weight,
    int hidden_dim,
    float epsilon
) {
    // One block per row
    const int row = blockIdx.x;
    const int tid = threadIdx.x;

    // Pointers to this row
    const float* row_input = input + row * hidden_dim;
    float* row_output = output + row * hidden_dim;

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
            float4 vec = *reinterpret_cast<const float4*>(&row_input[global_idx]);

            local_values[i + 0] = vec.x;
            local_values[i + 1] = vec.y;
            local_values[i + 2] = vec.z;
            local_values[i + 3] = vec.w;

            sum_sq += vec.x * vec.x + vec.y * vec.y +
                      vec.z * vec.z + vec.w * vec.w;
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
    __shared__ float warp_sums[NUM_WARPS];

    int lane = tid % WARP_SIZE;
    int warp_id = tid / WARP_SIZE;

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
            float4 w = *reinterpret_cast<const float4*>(&weight[global_idx]);

            // Compute normalized output
            float4 out;
            out.x = local_values[i + 0] * rrms * w.x;
            out.y = local_values[i + 1] * rrms * w.y;
            out.z = local_values[i + 2] * rrms * w.z;
            out.w = local_values[i + 3] * rrms * w.w;

            // Store (vectorized)
            *reinterpret_cast<float4*>(&row_output[global_idx]) = out;
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
void rmsnorm_forward_optimized(
    float* output,
    const float* input,
    const float* weight,
    int num_tokens,
    int hidden_dim,
    float epsilon,
    cudaStream_t stream
) {
    // Choose configuration based on hidden_dim
    // Goal: Each thread handles multiple elements, using all registers

    if (hidden_dim <= 1024) {
        // Small hidden dim: 256 threads, 4 elements each
        constexpr int BLOCK_SIZE = 256;
        constexpr int ELEMENTS_PER_THREAD = 4;

        dim3 grid(num_tokens);
        dim3 block(BLOCK_SIZE);

        rmsnorm_optimized_kernel<BLOCK_SIZE, ELEMENTS_PER_THREAD>
            <<<grid, block, 0, stream>>>(
                output, input, weight, hidden_dim, epsilon
            );
    } else if (hidden_dim <= 4096) {
        // Medium hidden dim: 256 threads, 16 elements each (covers 4096)
        constexpr int BLOCK_SIZE = 256;
        constexpr int ELEMENTS_PER_THREAD = 16;

        dim3 grid(num_tokens);
        dim3 block(BLOCK_SIZE);

        rmsnorm_optimized_kernel<BLOCK_SIZE, ELEMENTS_PER_THREAD>
            <<<grid, block, 0, stream>>>(
                output, input, weight, hidden_dim, epsilon
            );
    } else {
        // Large hidden dim: 512 threads, 16 elements each (covers 8192)
        constexpr int BLOCK_SIZE = 512;
        constexpr int ELEMENTS_PER_THREAD = 16;

        dim3 grid(num_tokens);
        dim3 block(BLOCK_SIZE);

        rmsnorm_optimized_kernel<BLOCK_SIZE, ELEMENTS_PER_THREAD>
            <<<grid, block, 0, stream>>>(
                output, input, weight, hidden_dim, epsilon
            );
    }

    CUDA_CHECK_LAST();
}
```

---

## Fused RMSNorm Kernel

For even better performance, we can fuse RMSNorm with other operations:

Add to `rmsnorm.cu`:

```cuda
// =============================================================================
// Fused RMSNorm + Residual Addition
// =============================================================================

/**
 * Common pattern in transformers:
 *   output = RMSNorm(input + residual)
 *
 * Fusing saves one memory round-trip!
 */
template<int BLOCK_SIZE, int ELEMENTS_PER_THREAD>
__global__ void rmsnorm_residual_kernel(
    float* __restrict__ output,          // [num_tokens, hidden_dim]
    float* __restrict__ residual_out,    // [num_tokens, hidden_dim] - updated residual
    const float* __restrict__ input,     // [num_tokens, hidden_dim]
    const float* __restrict__ residual,  // [num_tokens, hidden_dim]
    const float* __restrict__ weight,    // [hidden_dim]
    int hidden_dim,
    float epsilon
) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x;

    const float* row_input = input + row * hidden_dim;
    const float* row_residual = residual + row * hidden_dim;
    float* row_output = output + row * hidden_dim;
    float* row_residual_out = residual_out + row * hidden_dim;

    // Cache for input + residual
    float local_values[ELEMENTS_PER_THREAD];
    float sum_sq = 0.0f;

    // Load and add residual
    #pragma unroll
    for (int i = 0; i < ELEMENTS_PER_THREAD; i += 4) {
        int idx = tid * ELEMENTS_PER_THREAD + i;

        if (idx + 3 < hidden_dim) {
            float4 in = *reinterpret_cast<const float4*>(&row_input[idx]);
            float4 res = *reinterpret_cast<const float4*>(&row_residual[idx]);

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
    __shared__ float warp_sums[NUM_WARPS];

    int lane = tid % WARP_SIZE;
    int warp_id = tid / WARP_SIZE;

    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        sum_sq += __shfl_down_sync(0xffffffff, sum_sq, offset);
    }

    if (lane == 0) warp_sums[warp_id] = sum_sq;
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
            float4 w = *reinterpret_cast<const float4*>(&weight[idx]);

            // Normalized output
            float4 out;
            out.x = local_values[i + 0] * rrms * w.x;
            out.y = local_values[i + 1] * rrms * w.y;
            out.z = local_values[i + 2] * rrms * w.z;
            out.w = local_values[i + 3] * rrms * w.w;
            *reinterpret_cast<float4*>(&row_output[idx]) = out;

            // Updated residual (pre-normalized values)
            float4 res_out;
            res_out.x = local_values[i + 0];
            res_out.y = local_values[i + 1];
            res_out.z = local_values[i + 2];
            res_out.w = local_values[i + 3];
            *reinterpret_cast<float4*>(&row_residual_out[idx]) = res_out;
        }
    }
}
```

---

## FP16 Implementation

For production, we need FP16 support:

Add to `rmsnorm.cu`:

```cuda
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
template<int BLOCK_SIZE, int ELEMENTS_PER_THREAD>
__global__ void rmsnorm_fp16_kernel(
    half* __restrict__ output,
    const half* __restrict__ input,
    const half* __restrict__ weight,
    int hidden_dim,
    float epsilon
) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x;

    const half* row_input = input + row * hidden_dim;
    half* row_output = output + row * hidden_dim;

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
            half2 h01 = *reinterpret_cast<const half2*>(&row_input[idx]);
            half2 h23 = *reinterpret_cast<const half2*>(&row_input[idx + 2]);

            // Convert to float
            float2 f01 = __half22float2(h01);
            float2 f23 = __half22float2(h23);

            local_values[i + 0] = f01.x;
            local_values[i + 1] = f01.y;
            local_values[i + 2] = f23.x;
            local_values[i + 3] = f23.y;

            // Accumulate in FP32
            sum_sq += f01.x * f01.x + f01.y * f01.y +
                      f23.x * f23.x + f23.y * f23.y;
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

    if (lane == 0) warp_sums[warp_id] = sum_sq;
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
            half2 w01 = *reinterpret_cast<const half2*>(&weight[idx]);
            half2 w23 = *reinterpret_cast<const half2*>(&weight[idx + 2]);
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

            *reinterpret_cast<half2*>(&row_output[idx]) = h_out01;
            *reinterpret_cast<half2*>(&row_output[idx + 2]) = h_out23;
        }
    }
}

// FP16 wrapper
void rmsnorm_forward_fp16(
    half* output,
    const half* input,
    const half* weight,
    int num_tokens,
    int hidden_dim,
    float epsilon,
    cudaStream_t stream
) {
    constexpr int BLOCK_SIZE = 256;
    constexpr int ELEMENTS_PER_THREAD = 16;

    dim3 grid(num_tokens);
    dim3 block(BLOCK_SIZE);

    rmsnorm_fp16_kernel<BLOCK_SIZE, ELEMENTS_PER_THREAD>
        <<<grid, block, 0, stream>>>(
            output, input, weight, hidden_dim, epsilon
        );

    CUDA_CHECK_LAST();
}
```

---

## Testing and Verification

Create file: `mini_vllm/tests/cpp/test_rmsnorm.cu`

```cuda
// =============================================================================
// test_rmsnorm.cu - RMSNorm Unit Tests
// =============================================================================

#include "rmsnorm.cuh"
#include <cuda_runtime.h>
#include <cstdio>
#include <cmath>
#include <vector>
#include <random>

using namespace mini_vllm;

// CPU reference implementation
void rmsnorm_cpu(
    float* output,
    const float* input,
    const float* weight,
    int num_tokens,
    int hidden_dim,
    float epsilon
) {
    for (int row = 0; row < num_tokens; row++) {
        const float* row_in = input + row * hidden_dim;
        float* row_out = output + row * hidden_dim;

        // Compute sum of squares
        float sum_sq = 0.0f;
        for (int i = 0; i < hidden_dim; i++) {
            sum_sq += row_in[i] * row_in[i];
        }

        // Compute RMS
        float rms = sqrtf(sum_sq / hidden_dim + epsilon);
        float rrms = 1.0f / rms;

        // Normalize
        for (int i = 0; i < hidden_dim; i++) {
            row_out[i] = row_in[i] * rrms * weight[i];
        }
    }
}

// Calculate max absolute error
float max_abs_error(const float* a, const float* b, int n) {
    float max_err = 0.0f;
    for (int i = 0; i < n; i++) {
        float err = fabsf(a[i] - b[i]);
        if (err > max_err) max_err = err;
    }
    return max_err;
}

int main() {
    printf("=== RMSNorm Kernel Tests ===\n\n");

    // Test configuration
    const int num_tokens = 128;
    const int hidden_dim = 4096;
    const float epsilon = 1e-6f;

    const size_t size = num_tokens * hidden_dim * sizeof(float);
    const size_t weight_size = hidden_dim * sizeof(float);

    // Allocate host memory
    std::vector<float> h_input(num_tokens * hidden_dim);
    std::vector<float> h_weight(hidden_dim);
    std::vector<float> h_output_cpu(num_tokens * hidden_dim);
    std::vector<float> h_output_gpu(num_tokens * hidden_dim);

    // Initialize with random values
    std::mt19937 gen(42);
    std::normal_distribution<float> dist(0.0f, 1.0f);

    for (auto& v : h_input) v = dist(gen);
    for (auto& v : h_weight) v = dist(gen) * 0.1f + 1.0f;  // ~1.0

    // Allocate device memory
    float *d_input, *d_weight, *d_output;
    cudaMalloc(&d_input, size);
    cudaMalloc(&d_weight, weight_size);
    cudaMalloc(&d_output, size);

    // Copy to device
    cudaMemcpy(d_input, h_input.data(), size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_weight, h_weight.data(), weight_size, cudaMemcpyHostToDevice);

    // =======================================================================
    // Test 1: Correctness
    // =======================================================================
    printf("Test 1: Correctness\n");

    // CPU reference
    rmsnorm_cpu(h_output_cpu.data(), h_input.data(), h_weight.data(),
                num_tokens, hidden_dim, epsilon);

    // GPU
    rmsnorm_forward(d_output, d_input, d_weight, num_tokens, hidden_dim, epsilon);
    cudaDeviceSynchronize();

    // Copy back
    cudaMemcpy(h_output_gpu.data(), d_output, size, cudaMemcpyDeviceToHost);

    // Compare
    float max_err = max_abs_error(h_output_cpu.data(), h_output_gpu.data(),
                                   num_tokens * hidden_dim);

    printf("  Max absolute error: %.2e\n", max_err);
    if (max_err < 1e-5f) {
        printf("  [PASS] Results match within tolerance\n");
    } else {
        printf("  [FAIL] Error too large!\n");
    }

    // =======================================================================
    // Test 2: Performance
    // =======================================================================
    printf("\nTest 2: Performance\n");

    // Warmup
    for (int i = 0; i < 10; i++) {
        rmsnorm_forward(d_output, d_input, d_weight, num_tokens, hidden_dim, epsilon);
    }
    cudaDeviceSynchronize();

    // Benchmark
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    const int iterations = 1000;

    cudaEventRecord(start);
    for (int i = 0; i < iterations; i++) {
        rmsnorm_forward(d_output, d_input, d_weight, num_tokens, hidden_dim, epsilon);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    float avg_ms = ms / iterations;

    // Calculate bandwidth
    // Read: input (N*H) + weight (H)
    // Write: output (N*H)
    float bytes = (2.0f * num_tokens * hidden_dim + hidden_dim) * sizeof(float) * iterations;
    float bandwidth_gb = bytes / (ms * 1e6f);

    printf("  Time per call: %.3f us\n", avg_ms * 1000);
    printf("  Bandwidth: %.1f GB/s\n", bandwidth_gb);
    printf("  Tokens/sec: %.1f M\n", (num_tokens * iterations) / (ms * 1e3f));

    // =======================================================================
    // Test 3: Edge cases
    // =======================================================================
    printf("\nTest 3: Edge cases\n");

    // Test with different hidden dimensions
    int test_dims[] = {256, 512, 1024, 2048, 4096, 8192};
    for (int dim : test_dims) {
        std::vector<float> small_input(num_tokens * dim);
        std::vector<float> small_weight(dim);
        std::vector<float> small_cpu(num_tokens * dim);
        std::vector<float> small_gpu(num_tokens * dim);

        for (auto& v : small_input) v = dist(gen);
        for (auto& v : small_weight) v = 1.0f;

        float *d_si, *d_sw, *d_so;
        cudaMalloc(&d_si, num_tokens * dim * sizeof(float));
        cudaMalloc(&d_sw, dim * sizeof(float));
        cudaMalloc(&d_so, num_tokens * dim * sizeof(float));

        cudaMemcpy(d_si, small_input.data(), num_tokens * dim * sizeof(float),
                   cudaMemcpyHostToDevice);
        cudaMemcpy(d_sw, small_weight.data(), dim * sizeof(float),
                   cudaMemcpyHostToDevice);

        rmsnorm_cpu(small_cpu.data(), small_input.data(), small_weight.data(),
                    num_tokens, dim, epsilon);
        rmsnorm_forward(d_so, d_si, d_sw, num_tokens, dim, epsilon);
        cudaDeviceSynchronize();

        cudaMemcpy(small_gpu.data(), d_so, num_tokens * dim * sizeof(float),
                   cudaMemcpyDeviceToHost);

        float err = max_abs_error(small_cpu.data(), small_gpu.data(), num_tokens * dim);
        printf("  hidden_dim=%d: max_err=%.2e %s\n", dim, err,
               err < 1e-5f ? "[PASS]" : "[FAIL]");

        cudaFree(d_si); cudaFree(d_sw); cudaFree(d_so);
    }

    // Cleanup
    cudaFree(d_input);
    cudaFree(d_weight);
    cudaFree(d_output);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    printf("\n=== Tests Complete ===\n");
    return 0;
}
```

### Compilation and Running

```bash
# Compile the test
cd ~/work/mini_vllm
nvcc -O3 -std=c++17 \
    -I csrc/include \
    -o test_rmsnorm \
    tests/cpp/test_rmsnorm.cu \
    csrc/kernels/rmsnorm.cu

# Run the test
./test_rmsnorm
```

Expected output:

```
=== RMSNorm Kernel Tests ===

Test 1: Correctness
  Max absolute error: 1.19e-06
  [PASS] Results match within tolerance

Test 2: Performance
  Time per call: 12.345 us
  Bandwidth: 425.3 GB/s
  Tokens/sec: 10.4 M

Test 3: Edge cases
  hidden_dim=256: max_err=5.96e-07 [PASS]
  hidden_dim=512: max_err=7.15e-07 [PASS]
  hidden_dim=1024: max_err=8.34e-07 [PASS]
  hidden_dim=2048: max_err=9.53e-07 [PASS]
  hidden_dim=4096: max_err=1.19e-06 [PASS]
  hidden_dim=8192: max_err=1.43e-06 [PASS]

=== Tests Complete ===
```

---

## Summary

You've implemented RMSNorm with:

| Version       | Feature                            | Benefit                |
| ------------- | ---------------------------------- | ---------------------- |
| **Naive**     | Simple, readable                   | Good for learning      |
| **Optimized** | Register caching, vectorized loads | 2-3x faster            |
| **Fused**     | Combined with residual             | Reduced memory traffic |
| **FP16**      | Half precision                     | 2x memory bandwidth    |

### Key Optimizations

1. **Single pass**: Cache input in registers during reduction
2. **Vectorized access**: Use `float4`/`half2` for 128-bit transactions
3. **Warp shuffle**: Avoid shared memory for intra-warp reduction
4. **Fusion**: Combine with adjacent operations

---

## What's Next

Next, we'll implement **RoPE (Rotary Position Embeddings)**, which:

- Encodes position information into queries and keys
- Uses trigonometric functions for rotation
- Requires efficient batched computation

Continue to: [04_rope_kernel.md](./04_rope_kernel.md)
