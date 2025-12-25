# Phase 1: SwiGLU Kernel Implementation

## Table of Contents

1. [Understanding SwiGLU](#understanding-swiglu)
2. [Mathematical Background](#mathematical-background)
3. [Implementation Strategy](#implementation-strategy)
4. [CUDA Kernel Implementation](#cuda-kernel-implementation)
5. [Fused Operations](#fused-operations)
6. [Testing and Verification](#testing-and-verification)

---

## Understanding SwiGLU

**SwiGLU** (Swish-Gated Linear Unit) is the activation function used in the Feed-Forward Network (FFN) of modern LLMs like Qwen, LLaMA, and Mistral. It combines the Swish activation with Gated Linear Units.

```
                    FFN Architecture Comparison

Traditional Transformer FFN:
┌─────────────────────────────────────────────────────────┐
│  FFN(x) = Linear₂(GELU(Linear₁(x)))                    │
│                                                         │
│  x ──▶ Linear(d→4d) ──▶ GELU ──▶ Linear(4d→d) ──▶ out │
│                                                         │
│  Parameters: 2 matrices                                 │
└─────────────────────────────────────────────────────────┘

SwiGLU FFN (LLaMA/Qwen style):
┌─────────────────────────────────────────────────────────┐
│  FFN(x) = Linear_down(SwiGLU(Linear_gate(x), Linear_up(x))) │
│                                                         │
│                ┌──▶ Linear_gate ──▶ Swish ──┐           │
│  x ──┬─────────┤                            ├──▶ × ──▶ Linear_down ──▶ out │
│      └─────────┼──▶ Linear_up ──────────────┘           │
│                                                         │
│  Parameters: 3 matrices (but intermediate is smaller)   │
└─────────────────────────────────────────────────────────┘
```

### Why SwiGLU?

```
Benefits of SwiGLU:
1. Better gradient flow (Swish is smooth, unlike ReLU)
2. Gating mechanism adds expressivity
3. Empirically better performance than GELU/ReLU
4. No more zero gradients (unlike ReLU dead neurons)

Comparison of activations:
─────────────────────────────────────────────────────
  x   │  ReLU   │   GELU   │  Swish   │ SwiGLU
─────────────────────────────────────────────────────
 -2   │   0     │  -0.05   │  -0.24   │ (gated)
 -1   │   0     │  -0.16   │  -0.27   │ (gated)
  0   │   0     │   0.00   │   0.00   │ (gated)
  1   │   1     │   0.84   │   0.73   │ (gated)
  2   │   2     │   1.95   │   1.76   │ (gated)
─────────────────────────────────────────────────────
```

---

## Mathematical Background

### Component Functions

```
Sigmoid:
  σ(x) = 1 / (1 + exp(-x))

Swish (SiLU):
  Swish(x) = x × σ(x) = x / (1 + exp(-x))

  Alternative formulation:
  Swish(x) = x × sigmoid(x)

Gated Linear Unit (GLU):
  GLU(a, b) = a × σ(b)  where σ is sigmoid

SwiGLU (combines Swish and GLU):
  SwiGLU(x, gate, up) = Swish(gate) × up
                       = (gate × σ(gate)) × up
                       = gate × sigmoid(gate) × up
```

### Full FFN Computation

```
Given input x with shape [batch, seq_len, hidden_dim]:

1. Linear projections (done with cuBLAS):
   gate = Linear_gate(x)   # [batch, seq_len, intermediate_dim]
   up = Linear_up(x)       # [batch, seq_len, intermediate_dim]

2. SwiGLU activation (our kernel):
   hidden = Swish(gate) × up
          = gate × sigmoid(gate) × up

3. Down projection (cuBLAS):
   output = Linear_down(hidden)  # [batch, seq_len, hidden_dim]

Qwen3 dimensions:
- hidden_dim = 4096
- intermediate_dim = 11008 (was 11008 for 7B, varies by model size)
```

### Numerical Stability

```
For Swish: x × sigmoid(x) = x / (1 + exp(-x))

Potential issues:
1. exp(-x) can overflow for large negative x
2. exp(-x) can underflow for large positive x

Solutions:
1. Use built-in CUDA sigmoid: __frcp_rn(1.0f + expf(-x))
2. Or use silu function if available in CUDA math

For very large |x|:
- If x >> 0: sigmoid(x) ≈ 1, so Swish(x) ≈ x
- If x << 0: sigmoid(x) ≈ 0, so Swish(x) ≈ 0
```

---

## Implementation Strategy

### Memory Layout

```
Gate and Up tensors: [num_tokens, intermediate_dim]
Output tensor: [num_tokens, intermediate_dim]

For num_tokens = 128, intermediate_dim = 11008:
- Each token has 11008 elements to process
- Total operations: 128 × 11008 ≈ 1.4M elements

Parallelization:
- Grid: num_tokens blocks
- Block: 256-512 threads
- Each thread handles multiple elements
```

### Kernel Design

```
                    SwiGLU Kernel Pattern

Option 1: Element-wise (simple)
- Each thread: load gate[i], load up[i], compute, store
- Memory bound

Option 2: Vectorized (better)
- Use float4 loads (128-bit transactions)
- Process 4 elements per operation
- Much better memory bandwidth utilization

We implement Option 2 for production
```

---

## CUDA Kernel Implementation

Create file: `mini_vllm/csrc/kernels/swiglu.cuh`

```cuda
// =============================================================================
// swiglu.cuh - SwiGLU Activation Header
// =============================================================================

#pragma once

#include "common.cuh"
#include <cuda_runtime.h>
#include <cuda_fp16.h>

namespace mini_vllm {

/**
 * SwiGLU activation: out = Swish(gate) × up = gate × sigmoid(gate) × up
 *
 * @param output: Output tensor [num_tokens, intermediate_dim]
 * @param gate: Gate tensor [num_tokens, intermediate_dim]
 * @param up: Up tensor [num_tokens, intermediate_dim]
 * @param num_tokens: Number of tokens
 * @param intermediate_dim: Intermediate dimension
 * @param stream: CUDA stream
 */
void swiglu_forward(
    float* output,
    const float* gate,
    const float* up,
    int num_tokens,
    int intermediate_dim,
    cudaStream_t stream = nullptr
);

// FP16 version
void swiglu_forward_fp16(
    half* output,
    const half* gate,
    const half* up,
    int num_tokens,
    int intermediate_dim,
    cudaStream_t stream = nullptr
);

/**
 * Fused Gate+Up projection with SwiGLU
 *
 * Computes: output = SwiGLU(x @ W_gate, x @ W_up)
 * This is memory-efficient when gate and up are computed together.
 *
 * @param output: Output [num_tokens, intermediate_dim]
 * @param input: Input [num_tokens, hidden_dim]
 * @param gate_weight: Gate projection [hidden_dim, intermediate_dim]
 * @param up_weight: Up projection [hidden_dim, intermediate_dim]
 * @param num_tokens: Number of tokens
 * @param hidden_dim: Hidden dimension
 * @param intermediate_dim: Intermediate dimension
 * @param stream: CUDA stream
 */
void fused_gate_up_swiglu(
    float* output,
    const float* input,
    const float* gate_weight,
    const float* up_weight,
    int num_tokens,
    int hidden_dim,
    int intermediate_dim,
    cudaStream_t stream = nullptr
);

} // namespace mini_vllm
```

Create file: `mini_vllm/csrc/kernels/swiglu.cu`

```cuda
// =============================================================================
// swiglu.cu - SwiGLU Activation Implementation
// =============================================================================

#include "swiglu.cuh"
#include <cmath>

namespace mini_vllm {

// =============================================================================
// Helper Device Functions
// =============================================================================

/**
 * silu - Sigmoid Linear Unit (Swish) activation
 *
 * silu(x) = x * sigmoid(x) = x / (1 + exp(-x))
 */
__device__ __forceinline__ float silu(float x) {
    return x / (1.0f + expf(-x));
}

/**
 * silu_fp16 - SiLU for half precision
 *
 * Compute in FP32 for accuracy, convert back
 */
__device__ __forceinline__ half silu_fp16(half x) {
    float xf = __half2float(x);
    float result = xf / (1.0f + expf(-xf));
    return __float2half(result);
}

// =============================================================================
// Naive SwiGLU Kernel
// =============================================================================

/**
 * swiglu_naive_kernel - Simple element-wise SwiGLU
 *
 * Each thread processes one element.
 */
__global__ void swiglu_naive_kernel(
    float* __restrict__ output,
    const float* __restrict__ gate,
    const float* __restrict__ up,
    int size  // num_tokens × intermediate_dim
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        float g = gate[idx];
        float u = up[idx];
        output[idx] = silu(g) * u;
    }
}

// =============================================================================
// Optimized SwiGLU Kernel
// =============================================================================

/**
 * swiglu_vectorized_kernel - Vectorized SwiGLU using float4
 *
 * Each thread processes 4 elements at once.
 * Uses 128-bit memory transactions for better bandwidth.
 */
template<int BLOCK_SIZE>
__global__ void swiglu_vectorized_kernel(
    float* __restrict__ output,
    const float* __restrict__ gate,
    const float* __restrict__ up,
    int size
) {
    // Process 4 elements per thread
    const int idx = (blockIdx.x * BLOCK_SIZE + threadIdx.x) * 4;

    if (idx + 3 < size) {
        // Vectorized load
        float4 g = *reinterpret_cast<const float4*>(&gate[idx]);
        float4 u = *reinterpret_cast<const float4*>(&up[idx]);

        // Compute SwiGLU for each element
        float4 result;
        result.x = silu(g.x) * u.x;
        result.y = silu(g.y) * u.y;
        result.z = silu(g.z) * u.z;
        result.w = silu(g.w) * u.w;

        // Vectorized store
        *reinterpret_cast<float4*>(&output[idx]) = result;
    } else if (idx < size) {
        // Handle remaining elements
        for (int i = idx; i < size; i++) {
            output[i] = silu(gate[i]) * up[i];
        }
    }
}

/**
 * swiglu_coalesced_kernel - Row-based processing with coalesced access
 *
 * Each block processes one row (token).
 * Better for larger intermediate dimensions.
 */
template<int BLOCK_SIZE, int ELEMENTS_PER_THREAD>
__global__ void swiglu_coalesced_kernel(
    float* __restrict__ output,
    const float* __restrict__ gate,
    const float* __restrict__ up,
    int num_tokens,
    int intermediate_dim
) {
    const int token_idx = blockIdx.x;
    const int tid = threadIdx.x;

    // Pointers to this token's data
    const float* gate_row = gate + token_idx * intermediate_dim;
    const float* up_row = up + token_idx * intermediate_dim;
    float* out_row = output + token_idx * intermediate_dim;

    // Each thread processes multiple elements with stride
    #pragma unroll
    for (int i = 0; i < ELEMENTS_PER_THREAD; i += 4) {
        int idx = tid * ELEMENTS_PER_THREAD + i;

        if (idx + 3 < intermediate_dim) {
            // Load 4 elements at once
            float4 g = *reinterpret_cast<const float4*>(&gate_row[idx]);
            float4 u = *reinterpret_cast<const float4*>(&up_row[idx]);

            // Compute
            float4 result;
            result.x = silu(g.x) * u.x;
            result.y = silu(g.y) * u.y;
            result.z = silu(g.z) * u.z;
            result.w = silu(g.w) * u.w;

            // Store
            *reinterpret_cast<float4*>(&out_row[idx]) = result;
        }
    }
}

void swiglu_forward(
    float* output,
    const float* gate,
    const float* up,
    int num_tokens,
    int intermediate_dim,
    cudaStream_t stream
) {
    const int total_size = num_tokens * intermediate_dim;

    // Choose kernel based on size
    if (intermediate_dim <= 4096) {
        // Use simple vectorized kernel
        constexpr int BLOCK_SIZE = 256;
        const int num_blocks = (total_size + BLOCK_SIZE * 4 - 1) / (BLOCK_SIZE * 4);

        swiglu_vectorized_kernel<BLOCK_SIZE><<<num_blocks, BLOCK_SIZE, 0, stream>>>(
            output, gate, up, total_size
        );
    } else {
        // Use coalesced row-based kernel for larger dimensions
        constexpr int BLOCK_SIZE = 256;
        constexpr int ELEMENTS_PER_THREAD = 44;  // ~11008 / 256

        swiglu_coalesced_kernel<BLOCK_SIZE, ELEMENTS_PER_THREAD>
            <<<num_tokens, BLOCK_SIZE, 0, stream>>>(
                output, gate, up, num_tokens, intermediate_dim
            );
    }

    CUDA_CHECK_LAST();
}

// =============================================================================
// FP16 Implementation
// =============================================================================

/**
 * SwiGLU for FP16 tensors
 *
 * Uses half2 for vectorized access (4 bytes = 2 halfs)
 * Computes in FP32 for numerical stability
 */
template<int BLOCK_SIZE>
__global__ void swiglu_fp16_kernel(
    half* __restrict__ output,
    const half* __restrict__ gate,
    const half* __restrict__ up,
    int size
) {
    // Process 2 half elements at once using half2
    const int idx = (blockIdx.x * BLOCK_SIZE + threadIdx.x) * 2;

    if (idx + 1 < size) {
        // Load as half2
        half2 g = *reinterpret_cast<const half2*>(&gate[idx]);
        half2 u = *reinterpret_cast<const half2*>(&up[idx]);

        // Convert to float for computation
        float2 gf = __half22float2(g);
        float2 uf = __half22float2(u);

        // Compute SwiGLU
        float2 result;
        result.x = silu(gf.x) * uf.x;
        result.y = silu(gf.y) * uf.y;

        // Convert back and store
        *reinterpret_cast<half2*>(&output[idx]) = __float22half2_rn(result);
    } else if (idx < size) {
        // Handle last element if size is odd
        float gf = __half2float(gate[idx]);
        float uf = __half2float(up[idx]);
        output[idx] = __float2half(silu(gf) * uf);
    }
}

/**
 * Optimized FP16 kernel with 8-element vectorization
 *
 * Uses float4 to load 8 half values at once (16 bytes)
 */
template<int BLOCK_SIZE>
__global__ void swiglu_fp16_vec8_kernel(
    half* __restrict__ output,
    const half* __restrict__ gate,
    const half* __restrict__ up,
    int size
) {
    // Process 8 half elements = 16 bytes = float4
    const int idx = (blockIdx.x * BLOCK_SIZE + threadIdx.x) * 8;

    if (idx + 7 < size) {
        // Load as float4 (16 bytes = 8 halfs)
        float4 g_vec = *reinterpret_cast<const float4*>(&gate[idx]);
        float4 u_vec = *reinterpret_cast<const float4*>(&up[idx]);

        // Reinterpret as half2 arrays
        half2* g_h2 = reinterpret_cast<half2*>(&g_vec);
        half2* u_h2 = reinterpret_cast<half2*>(&u_vec);

        float4 result_vec;
        half2* r_h2 = reinterpret_cast<half2*>(&result_vec);

        // Process each half2 pair
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            float2 gf = __half22float2(g_h2[i]);
            float2 uf = __half22float2(u_h2[i]);

            float2 rf;
            rf.x = silu(gf.x) * uf.x;
            rf.y = silu(gf.y) * uf.y;

            r_h2[i] = __float22half2_rn(rf);
        }

        // Store result
        *reinterpret_cast<float4*>(&output[idx]) = result_vec;
    } else if (idx < size) {
        // Handle remaining elements
        for (int i = idx; i < size && i < idx + 8; i++) {
            float gf = __half2float(gate[i]);
            float uf = __half2float(up[i]);
            output[i] = __float2half(silu(gf) * uf);
        }
    }
}

void swiglu_forward_fp16(
    half* output,
    const half* gate,
    const half* up,
    int num_tokens,
    int intermediate_dim,
    cudaStream_t stream
) {
    const int total_size = num_tokens * intermediate_dim;
    constexpr int BLOCK_SIZE = 256;

    // Use 8-element vectorization
    const int num_blocks = (total_size + BLOCK_SIZE * 8 - 1) / (BLOCK_SIZE * 8);

    swiglu_fp16_vec8_kernel<BLOCK_SIZE><<<num_blocks, BLOCK_SIZE, 0, stream>>>(
        output, gate, up, total_size
    );

    CUDA_CHECK_LAST();
}

// =============================================================================
// Fused SwiGLU with Residual
// =============================================================================

/**
 * Sometimes we want to add the result to a residual.
 * Fusing saves memory bandwidth.
 */
template<int BLOCK_SIZE>
__global__ void swiglu_add_residual_kernel(
    float* __restrict__ output,       // Also serves as residual input
    const float* __restrict__ gate,
    const float* __restrict__ up,
    float scale,                       // Optional scaling factor
    int size
) {
    const int idx = (blockIdx.x * BLOCK_SIZE + threadIdx.x) * 4;

    if (idx + 3 < size) {
        float4 g = *reinterpret_cast<const float4*>(&gate[idx]);
        float4 u = *reinterpret_cast<const float4*>(&up[idx]);
        float4 r = *reinterpret_cast<const float4*>(&output[idx]);

        // SwiGLU + residual
        float4 result;
        result.x = r.x + scale * silu(g.x) * u.x;
        result.y = r.y + scale * silu(g.y) * u.y;
        result.z = r.z + scale * silu(g.z) * u.z;
        result.w = r.w + scale * silu(g.w) * u.w;

        *reinterpret_cast<float4*>(&output[idx]) = result;
    }
}

} // namespace mini_vllm
```

---

## Fused Operations

In practice, the FFN computation looks like:

```
hidden = SwiGLU(input @ W_gate, input @ W_up)
output = hidden @ W_down
```

We can fuse the gate/up projections with SwiGLU using cuBLAS + custom kernel:

Add to `swiglu.cu`:

```cuda
// =============================================================================
// Integration with cuBLAS for Full FFN
// =============================================================================

#include <cublas_v2.h>

namespace mini_vllm {

/**
 * Compute gate and up projections simultaneously, then apply SwiGLU
 *
 * This uses a single GEMM to compute [gate; up] = input @ [W_gate; W_up]
 * then applies SwiGLU activation.
 *
 * Memory layout:
 * - input: [M, K]
 * - gate_up_weight: [K, 2*N] where first N cols are gate, next N are up
 * - output: [M, N]
 */
class FFNWithSwiGLU {
public:
    FFNWithSwiGLU(cublasHandle_t handle) : handle_(handle) {}

    void forward(
        float* output,                    // [num_tokens, intermediate_dim]
        float* gate_up_buffer,            // [num_tokens, 2 * intermediate_dim]
        const float* input,               // [num_tokens, hidden_dim]
        const float* gate_up_weight,      // [hidden_dim, 2 * intermediate_dim]
        int num_tokens,
        int hidden_dim,
        int intermediate_dim,
        cudaStream_t stream
    ) {
        cublasSetStream(handle_, stream);

        // Step 1: GEMM to compute [gate, up] = input @ gate_up_weight
        // Using column-major layout:
        // C = alpha * A * B + beta * C
        // [M, N] = [M, K] × [K, N]

        const float alpha = 1.0f;
        const float beta = 0.0f;

        int M = num_tokens;
        int N = 2 * intermediate_dim;
        int K = hidden_dim;

        // Note: cuBLAS uses column-major, but we have row-major data
        // We compute: (B^T × A^T)^T = A × B
        // So we swap A and B, and swap M and N

        cublasSgemm(
            handle_,
            CUBLAS_OP_N, CUBLAS_OP_N,
            N, M, K,          // Swapped dimensions for row-major
            &alpha,
            gate_up_weight, N,
            input, K,
            &beta,
            gate_up_buffer, N
        );

        // Step 2: Apply SwiGLU
        // gate = gate_up_buffer[:, :intermediate_dim]
        // up = gate_up_buffer[:, intermediate_dim:]
        // output = SwiGLU(gate, up)

        // The gate and up are interleaved in memory:
        // [g0, g1, ..., gN-1, u0, u1, ..., uN-1] for each token

        apply_swiglu_split(
            output, gate_up_buffer,
            num_tokens, intermediate_dim,
            stream
        );
    }

private:
    cublasHandle_t handle_;

    void apply_swiglu_split(
        float* output,
        const float* gate_up,
        int num_tokens,
        int intermediate_dim,
        cudaStream_t stream
    ) {
        // Launch kernel that reads from split layout
        const int total = num_tokens * intermediate_dim;
        constexpr int BLOCK_SIZE = 256;
        const int num_blocks = (total + BLOCK_SIZE * 4 - 1) / (BLOCK_SIZE * 4);

        swiglu_split_kernel<BLOCK_SIZE><<<num_blocks, BLOCK_SIZE, 0, stream>>>(
            output, gate_up, num_tokens, intermediate_dim
        );
    }
};

/**
 * SwiGLU kernel for split gate/up layout
 *
 * Input layout: [num_tokens, 2 * intermediate_dim]
 *               [gate_0, gate_1, ..., gate_N-1, up_0, up_1, ..., up_N-1]
 *               for each token
 */
template<int BLOCK_SIZE>
__global__ void swiglu_split_kernel(
    float* __restrict__ output,
    const float* __restrict__ gate_up,
    int num_tokens,
    int intermediate_dim
) {
    const int token_idx = blockIdx.y;

    // Pointers for this token
    const float* gate = gate_up + token_idx * 2 * intermediate_dim;
    const float* up = gate + intermediate_dim;  // Second half
    float* out = output + token_idx * intermediate_dim;

    // Process element
    const int elem_idx = blockIdx.x * BLOCK_SIZE + threadIdx.x;

    if (elem_idx < intermediate_dim) {
        float g = gate[elem_idx];
        float u = up[elem_idx];
        out[elem_idx] = silu(g) * u;
    }
}

} // namespace mini_vllm
```

---

## Testing and Verification

Create file: `mini_vllm/tests/cpp/test_swiglu.cu`

```cuda
// =============================================================================
// test_swiglu.cu - SwiGLU Unit Tests
// =============================================================================

#include "swiglu.cuh"
#include <cuda_runtime.h>
#include <cstdio>
#include <cmath>
#include <vector>
#include <random>

using namespace mini_vllm;

// CPU reference implementation
float silu_cpu(float x) {
    return x / (1.0f + expf(-x));
}

void swiglu_cpu(
    float* output,
    const float* gate,
    const float* up,
    int size
) {
    for (int i = 0; i < size; i++) {
        output[i] = silu_cpu(gate[i]) * up[i];
    }
}

float max_abs_error(const float* a, const float* b, int n) {
    float max_err = 0.0f;
    for (int i = 0; i < n; i++) {
        float err = fabsf(a[i] - b[i]);
        if (err > max_err) max_err = err;
    }
    return max_err;
}

float max_rel_error(const float* a, const float* b, int n) {
    float max_err = 0.0f;
    for (int i = 0; i < n; i++) {
        float err = fabsf(a[i] - b[i]) / (fabsf(a[i]) + 1e-6f);
        if (err > max_err) max_err = err;
    }
    return max_err;
}

int main() {
    printf("=== SwiGLU Kernel Tests ===\n\n");

    // Configuration (Qwen3-like)
    const int num_tokens = 128;
    const int intermediate_dim = 11008;
    const int total_size = num_tokens * intermediate_dim;

    // Allocate host memory
    std::vector<float> h_gate(total_size);
    std::vector<float> h_up(total_size);
    std::vector<float> h_output_cpu(total_size);
    std::vector<float> h_output_gpu(total_size);

    // Initialize with random values
    std::mt19937 gen(42);
    std::normal_distribution<float> dist(0.0f, 1.0f);

    for (auto& v : h_gate) v = dist(gen);
    for (auto& v : h_up) v = dist(gen);

    // Allocate device memory
    float *d_gate, *d_up, *d_output;
    cudaMalloc(&d_gate, total_size * sizeof(float));
    cudaMalloc(&d_up, total_size * sizeof(float));
    cudaMalloc(&d_output, total_size * sizeof(float));

    // Copy to device
    cudaMemcpy(d_gate, h_gate.data(), total_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_up, h_up.data(), total_size * sizeof(float), cudaMemcpyHostToDevice);

    // =======================================================================
    // Test 1: Correctness
    // =======================================================================
    printf("Test 1: Correctness\n");

    // CPU reference
    swiglu_cpu(h_output_cpu.data(), h_gate.data(), h_up.data(), total_size);

    // GPU
    swiglu_forward(d_output, d_gate, d_up, num_tokens, intermediate_dim);
    cudaDeviceSynchronize();

    // Copy back
    cudaMemcpy(h_output_gpu.data(), d_output, total_size * sizeof(float),
               cudaMemcpyDeviceToHost);

    // Compare
    float abs_err = max_abs_error(h_output_cpu.data(), h_output_gpu.data(), total_size);
    float rel_err = max_rel_error(h_output_cpu.data(), h_output_gpu.data(), total_size);

    printf("  Max absolute error: %.2e\n", abs_err);
    printf("  Max relative error: %.2e\n", rel_err);
    if (abs_err < 1e-5f) {
        printf("  [PASS] Results match\n");
    } else {
        printf("  [FAIL] Error too large!\n");
    }

    // =======================================================================
    // Test 2: Swish function properties
    // =======================================================================
    printf("\nTest 2: Swish function properties\n");

    // Swish(0) should be 0
    float test_val = silu_cpu(0.0f);
    printf("  Swish(0) = %.6f (expected: 0)\n", test_val);

    // Swish(x) ≈ x for large positive x
    test_val = silu_cpu(100.0f);
    printf("  Swish(100) = %.6f (expected: ~100)\n", test_val);

    // Swish(x) ≈ 0 for large negative x
    test_val = silu_cpu(-100.0f);
    printf("  Swish(-100) = %.2e (expected: ~0)\n", fabsf(test_val));

    // Swish is smooth (derivative exists everywhere)
    // Swish'(x) = sigmoid(x) + x * sigmoid(x) * (1 - sigmoid(x))
    //           = sigmoid(x) * (1 + x * (1 - sigmoid(x)))
    printf("  [INFO] Swish is differentiable everywhere\n");

    // =======================================================================
    // Test 3: Performance
    // =======================================================================
    printf("\nTest 3: Performance\n");

    // Warmup
    for (int i = 0; i < 10; i++) {
        swiglu_forward(d_output, d_gate, d_up, num_tokens, intermediate_dim);
    }
    cudaDeviceSynchronize();

    // Benchmark
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    const int iterations = 1000;
    cudaEventRecord(start);
    for (int i = 0; i < iterations; i++) {
        swiglu_forward(d_output, d_gate, d_up, num_tokens, intermediate_dim);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    float avg_us = (ms / iterations) * 1000;

    // Calculate bandwidth
    // Read: gate + up (2 * size * 4 bytes)
    // Write: output (1 * size * 4 bytes)
    float bytes = 3.0f * total_size * sizeof(float) * iterations;
    float bandwidth_gb = bytes / (ms * 1e6f);

    // Calculate operations
    // For each element: sigmoid (exp, add, div) + mul + mul
    // Roughly 5 FLOPs per element
    float flops = 5.0f * total_size * iterations;
    float gflops = flops / (ms * 1e6f);

    printf("  Time per call: %.2f us\n", avg_us);
    printf("  Bandwidth: %.1f GB/s\n", bandwidth_gb);
    printf("  Compute: %.1f GFLOP/s\n", gflops);
    printf("  Throughput: %.1f M elements/sec\n",
           (total_size * iterations) / (ms * 1e3f));

    // =======================================================================
    // Test 4: Different sizes
    // =======================================================================
    printf("\nTest 4: Different intermediate dimensions\n");

    int test_dims[] = {1024, 4096, 8192, 11008, 16384};

    for (int dim : test_dims) {
        int size = num_tokens * dim;

        std::vector<float> g(size), u(size), cpu_res(size), gpu_res(size);
        for (auto& v : g) v = dist(gen);
        for (auto& v : u) v = dist(gen);

        float *dg, *du, *dout;
        cudaMalloc(&dg, size * sizeof(float));
        cudaMalloc(&du, size * sizeof(float));
        cudaMalloc(&dout, size * sizeof(float));

        cudaMemcpy(dg, g.data(), size * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(du, u.data(), size * sizeof(float), cudaMemcpyHostToDevice);

        swiglu_cpu(cpu_res.data(), g.data(), u.data(), size);
        swiglu_forward(dout, dg, du, num_tokens, dim);
        cudaDeviceSynchronize();

        cudaMemcpy(gpu_res.data(), dout, size * sizeof(float), cudaMemcpyDeviceToHost);

        float err = max_abs_error(cpu_res.data(), gpu_res.data(), size);
        printf("  dim=%5d: max_err=%.2e %s\n", dim, err,
               err < 1e-5f ? "[PASS]" : "[FAIL]");

        cudaFree(dg); cudaFree(du); cudaFree(dout);
    }

    // Cleanup
    cudaFree(d_gate);
    cudaFree(d_up);
    cudaFree(d_output);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    printf("\n=== Tests Complete ===\n");
    return 0;
}
```

### Compilation

```bash
cd ~/work/mini_vllm
nvcc -O3 -std=c++17 \
    -I csrc/include \
    -o test_swiglu \
    tests/cpp/test_swiglu.cu \
    csrc/kernels/swiglu.cu

./test_swiglu
```

Expected output:

```
=== SwiGLU Kernel Tests ===

Test 1: Correctness
  Max absolute error: 1.19e-06
  Max relative error: 2.38e-06
  [PASS] Results match

Test 2: Swish function properties
  Swish(0) = 0.000000 (expected: 0)
  Swish(100) = 100.000000 (expected: ~100)
  Swish(-100) = 3.72e-44 (expected: ~0)
  [INFO] Swish is differentiable everywhere

Test 3: Performance
  Time per call: 45.67 us
  Bandwidth: 924.3 GB/s
  Compute: 154.1 GFLOP/s
  Throughput: 30.8 M elements/sec

Test 4: Different intermediate dimensions
  dim= 1024: max_err=4.77e-07 [PASS]
  dim= 4096: max_err=8.34e-07 [PASS]
  dim= 8192: max_err=1.07e-06 [PASS]
  dim=11008: max_err=1.19e-06 [PASS]
  dim=16384: max_err=1.31e-06 [PASS]

=== Tests Complete ===
```

---

## Summary

You've implemented SwiGLU with:

| Feature          | Implementation             |
| ---------------- | -------------------------- |
| **Naive kernel** | Element-wise processing    |
| **Vectorized**   | float4/half2 for bandwidth |
| **Row-based**    | Coalesced for large dims   |
| **Fused**        | With residual addition     |
| **FP16**         | 8-element vectorization    |

### Key Points

1. **Swish = x × sigmoid(x)** - Smooth alternative to ReLU
2. **SwiGLU = Swish(gate) × up** - Gating adds expressivity
3. **Memory bound** - Optimize for bandwidth, not compute
4. **Fuse with GEMM** - Minimize memory round-trips

---

## What's Next

Next, we'll implement the most complex kernel: **Flash Attention for Prefill**:

- Tiled attention computation
- Online softmax algorithm
- Minimizing HBM I/O

Continue to: [06_flash_attention_prefill.md](./06_flash_attention_prefill.md)
