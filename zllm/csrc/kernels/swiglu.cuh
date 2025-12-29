// =============================================================================
// swiglu.cuh - SwiGLU Activation Header
// =============================================================================

#pragma once

#include "../include/common.cuh"
#include <cuda_fp16.h>
#include <cuda_runtime.h>

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
void swiglu_forward(float *output, const float *gate, const float *up,
                    int num_tokens, int intermediate_dim,
                    cudaStream_t stream = nullptr);

// FP16 version
void swiglu_forward_fp16(half *output, const half *gate, const half *up,
                         int num_tokens, int intermediate_dim,
                         cudaStream_t stream = nullptr);

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
void fused_gate_up_swiglu(float *output, const float *input,
                          const float *gate_weight, const float *up_weight,
                          int num_tokens, int hidden_dim, int intermediate_dim,
                          cudaStream_t stream = nullptr);

// Forward declaration of kernel for split layout
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
                                    int num_tokens, int intermediate_dim);

} // namespace mini_vllm
