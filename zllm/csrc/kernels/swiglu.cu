// =============================================================================
// swiglu.cu - SwiGLU Activation Implementation
// =============================================================================

#include "../include/common.cuh"
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
__device__ __forceinline__ float silu(float x) { return x / (1.0f + expf(-x)); }

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
__global__ void swiglu_naive_kernel(float *__restrict__ output,
                                    const float *__restrict__ gate,
                                    const float *__restrict__ up,
                                    int size // num_tokens × intermediate_dim
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
template <int BLOCK_SIZE>
__global__ void swiglu_vectorized_kernel(float *__restrict__ output,
                                         const float *__restrict__ gate,
                                         const float *__restrict__ up,
                                         int size) {
  // Process 4 elements per thread
  const int idx = (blockIdx.x * BLOCK_SIZE + threadIdx.x) * 4;

  if (idx + 3 < size) {
    // Vectorized load
    float4 g = *reinterpret_cast<const float4 *>(&gate[idx]);
    float4 u = *reinterpret_cast<const float4 *>(&up[idx]);

    // Compute SwiGLU for each element
    float4 result;
    result.x = silu(g.x) * u.x;
    result.y = silu(g.y) * u.y;
    result.z = silu(g.z) * u.z;
    result.w = silu(g.w) * u.w;

    // Vectorized store
    *reinterpret_cast<float4 *>(&output[idx]) = result;
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
template <int BLOCK_SIZE, int ELEMENTS_PER_THREAD>
__global__ void swiglu_coalesced_kernel(float *__restrict__ output,
                                        const float *__restrict__ gate,
                                        const float *__restrict__ up,
                                        int num_tokens, int intermediate_dim) {
  const int token_idx = blockIdx.x;
  const int tid = threadIdx.x;

  // Pointers to this token's data
  const float *gate_row = gate + token_idx * intermediate_dim;
  const float *up_row = up + token_idx * intermediate_dim;
  float *out_row = output + token_idx * intermediate_dim;

// Each thread processes multiple elements with stride
#pragma unroll
  for (int i = 0; i < ELEMENTS_PER_THREAD; i += 4) {
    int idx = tid * ELEMENTS_PER_THREAD + i;

    if (idx + 3 < intermediate_dim) {
      // Load 4 elements at once
      float4 g = *reinterpret_cast<const float4 *>(&gate_row[idx]);
      float4 u = *reinterpret_cast<const float4 *>(&up_row[idx]);

      // Compute
      float4 result;
      result.x = silu(g.x) * u.x;
      result.y = silu(g.y) * u.y;
      result.z = silu(g.z) * u.z;
      result.w = silu(g.w) * u.w;

      // Store
      *reinterpret_cast<float4 *>(&out_row[idx]) = result;
    }
  }
}

void swiglu_forward(float *output, const float *gate, const float *up,
                    int num_tokens, int intermediate_dim, cudaStream_t stream) {
  const int total_size = num_tokens * intermediate_dim;

  // Choose kernel based on size
  if (intermediate_dim <= 4096) {
    // Use simple vectorized kernel
    constexpr int BLOCK_SIZE = 256;
    const int num_blocks = (total_size + BLOCK_SIZE * 4 - 1) / (BLOCK_SIZE * 4);

    swiglu_vectorized_kernel<BLOCK_SIZE>
        <<<num_blocks, BLOCK_SIZE, 0, stream>>>(output, gate, up, total_size);
  } else {
    // Use coalesced row-based kernel for larger dimensions
    constexpr int BLOCK_SIZE = 256;
    constexpr int ELEMENTS_PER_THREAD = 44; // ~11008 / 256

    swiglu_coalesced_kernel<BLOCK_SIZE, ELEMENTS_PER_THREAD>
        <<<num_tokens, BLOCK_SIZE, 0, stream>>>(output, gate, up, num_tokens,
                                                intermediate_dim);
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
template <int BLOCK_SIZE>
__global__ void swiglu_fp16_kernel(half *__restrict__ output,
                                   const half *__restrict__ gate,
                                   const half *__restrict__ up, int size) {
  // Process 2 half elements at once using half2
  const int idx = (blockIdx.x * BLOCK_SIZE + threadIdx.x) * 2;

  if (idx + 1 < size) {
    // Load as half2
    half2 g = *reinterpret_cast<const half2 *>(&gate[idx]);
    half2 u = *reinterpret_cast<const half2 *>(&up[idx]);

    // Convert to float for computation
    float2 gf = __half22float2(g);
    float2 uf = __half22float2(u);

    // Compute SwiGLU
    float2 result;
    result.x = silu(gf.x) * uf.x;
    result.y = silu(gf.y) * uf.y;

    // Convert back and store
    *reinterpret_cast<half2 *>(&output[idx]) = __float22half2_rn(result);
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
template <int BLOCK_SIZE>
__global__ void swiglu_fp16_vec8_kernel(half *__restrict__ output,
                                        const half *__restrict__ gate,
                                        const half *__restrict__ up, int size) {
  // Process 8 half elements = 16 bytes = float4
  const int idx = (blockIdx.x * BLOCK_SIZE + threadIdx.x) * 8;

  if (idx + 7 < size) {
    // Load as float4 (16 bytes = 8 halfs)
    float4 g_vec = *reinterpret_cast<const float4 *>(&gate[idx]);
    float4 u_vec = *reinterpret_cast<const float4 *>(&up[idx]);

    // Reinterpret as half2 arrays
    half2 *g_h2 = reinterpret_cast<half2 *>(&g_vec);
    half2 *u_h2 = reinterpret_cast<half2 *>(&u_vec);

    float4 result_vec;
    half2 *r_h2 = reinterpret_cast<half2 *>(&result_vec);

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
    *reinterpret_cast<float4 *>(&output[idx]) = result_vec;
  } else if (idx < size) {
    // Handle remaining elements
    for (int i = idx; i < size && i < idx + 8; i++) {
      float gf = __half2float(gate[i]);
      float uf = __half2float(up[i]);
      output[i] = __float2half(silu(gf) * uf);
    }
  }
}

void swiglu_forward_fp16(half *output, const half *gate, const half *up,
                         int num_tokens, int intermediate_dim,
                         cudaStream_t stream) {
  const int total_size = num_tokens * intermediate_dim;
  constexpr int BLOCK_SIZE = 256;

  // Use 8-element vectorization
  const int num_blocks = (total_size + BLOCK_SIZE * 8 - 1) / (BLOCK_SIZE * 8);

  swiglu_fp16_vec8_kernel<BLOCK_SIZE>
      <<<num_blocks, BLOCK_SIZE, 0, stream>>>(output, gate, up, total_size);

  CUDA_CHECK_LAST();
}

// =============================================================================
// Fused SwiGLU with Residual
// =============================================================================

/**
 * Sometimes we want to add the result to a residual.
 * Fusing saves memory bandwidth.
 */
template <int BLOCK_SIZE>
__global__ void swiglu_add_residual_kernel(
    float *__restrict__ output, // Also serves as residual input
    const float *__restrict__ gate, const float *__restrict__ up,
    float scale, // Optional scaling factor
    int size) {
  const int idx = (blockIdx.x * BLOCK_SIZE + threadIdx.x) * 4;

  if (idx + 3 < size) {
    float4 g = *reinterpret_cast<const float4 *>(&gate[idx]);
    float4 u = *reinterpret_cast<const float4 *>(&up[idx]);
    float4 r = *reinterpret_cast<const float4 *>(&output[idx]);

    // SwiGLU + residual
    float4 result;
    result.x = r.x + scale * silu(g.x) * u.x;
    result.y = r.y + scale * silu(g.y) * u.y;
    result.z = r.z + scale * silu(g.z) * u.z;
    result.w = r.w + scale * silu(g.w) * u.w;

    *reinterpret_cast<float4 *>(&output[idx]) = result;
  }
}

} // namespace mini_vllm

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

  void
  forward(float *output,               // [num_tokens, intermediate_dim]
          float *gate_up_buffer,       // [num_tokens, 2 * intermediate_dim]
          const float *input,          // [num_tokens, hidden_dim]
          const float *gate_up_weight, // [hidden_dim, 2 * intermediate_dim]
          int num_tokens, int hidden_dim, int intermediate_dim,
          cudaStream_t stream) {
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

    cublasSgemm(handle_, CUBLAS_OP_N, CUBLAS_OP_N, N, M,
                K, // Swapped dimensions for row-major
                &alpha, gate_up_weight, N, input, K, &beta, gate_up_buffer, N);

    // Step 2: Apply SwiGLU
    // gate = gate_up_buffer[:, :intermediate_dim]
    // up = gate_up_buffer[:, intermediate_dim:]
    // output = SwiGLU(gate, up)

    // The gate and up are interleaved in memory:
    // [g0, g1, ..., gN-1, u0, u1, ..., uN-1] for each token

    apply_swiglu_split(output, gate_up_buffer, num_tokens, intermediate_dim,
                       stream);
  }

private:
  cublasHandle_t handle_;

  void apply_swiglu_split(float *output, const float *gate_up, int num_tokens,
                          int intermediate_dim, cudaStream_t stream) {
    // Launch kernel that reads from split layout
    const int total = num_tokens * intermediate_dim;
    constexpr int BLOCK_SIZE = 256;
    const int num_blocks = (total + BLOCK_SIZE * 4 - 1) / (BLOCK_SIZE * 4);

    swiglu_split_kernel<BLOCK_SIZE><<<num_blocks, BLOCK_SIZE, 0, stream>>>(
        output, gate_up, num_tokens, intermediate_dim);
  }
};

/**
 * SwiGLU kernel for split gate/up layout
 *
 * Input layout: [num_tokens, 2 * intermediate_dim]
 *               [gate_0, gate_1, ..., gate_N-1, up_0, up_1, ..., up_N-1]
 *               for each token
 */
template <int BLOCK_SIZE>
__global__ void swiglu_split_kernel(float *__restrict__ output,
                                    const float *__restrict__ gate_up,
                                    int num_tokens, int intermediate_dim) {
  const int token_idx = blockIdx.y;

  // Pointers for this token
  const float *gate = gate_up + token_idx * 2 * intermediate_dim;
  const float *up = gate + intermediate_dim; // Second half
  float *out = output + token_idx * intermediate_dim;

  // Process element
  const int elem_idx = blockIdx.x * BLOCK_SIZE + threadIdx.x;

  if (elem_idx < intermediate_dim) {
    float g = gate[elem_idx];
    float u = up[elem_idx];
    out[elem_idx] = silu(g) * u;
  }
}

} // namespace mini_vllm
