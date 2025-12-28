#pragma once
#include "../include/common.cuh"
#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace mini_vllm {
void rope_init_tables(float *cos_table, float *sin_table, int max_seq_len,
                      int head_dim, float theta_base,
                      cudaStream_t stream = nullptr);

void rope_forward(float *query, float *key, const float *cos_table,
                  const float *sin_table, const int *positions, int num_tokens,
                  int num_heads, int num_kv_heads, int head_dim,
                  cudaStream_t stream = nullptr);

void rope_forward_fp16(half *query, half *key, const float *cos_table,
                       const float *sin_table, const int *positions,
                       int num_tokens, int num_heads, int num_kv_heads,
                       int head_dim, cudaStream_t stream = nullptr);

} // namespace mini_vllm
