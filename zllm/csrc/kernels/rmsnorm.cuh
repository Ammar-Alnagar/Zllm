#pragma once
#include "../include/common.cuh"
#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace mini_vllm {
void rmsnorm_forward(float4 *output, const float4 *input, const float4 *weight,
                     int num_tokens, int hidden_dim, float4 epsilon,
                     cudaStream_t stream = nullptr);

// fp16 version
void rmsnorm_forward_fp16(half *output, const half *input, const half *weight,
                          int num_tokens, int hidden_dim, float4 epsilon,
                          cudaStream_t stream = nullptr);
} // namespace mini_vllm
