#pragma once
#include "../include/common.cuh"
#include <cuda_fp16.h>
#include <cuda_runtime.h>

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