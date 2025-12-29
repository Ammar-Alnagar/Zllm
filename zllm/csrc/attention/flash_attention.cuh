// =============================================================================
// flash_attention.cuh - Flash Attention Header
// =============================================================================

#pragma once

#include "../include/common.cuh"
#include <cuda_fp16.h>

namespace mini_vllm {

// =============================================================================
// Flash Attention Prefill Configuration
// =============================================================================

struct FlashAttentionConfig {
  int block_size;          // Tile size for tiled computation
  int num_heads;           // Number of attention heads
  int num_kv_heads;        // Number of KV heads (for GQA)
  int head_dim;            // Dimension of each head
  int max_seq_len;         // Maximum sequence length
  float softmax_scale;     // Softmax scaling factor (1/sqrt(head_dim))
  bool is_causal;          // Whether to apply causal masking
  int dtype_size;          // Size of data type (2 for FP16, 4 for FP32)
};

// =============================================================================
// Flash Attention Prefill Functions
// =============================================================================

// Forward pass for flash attention prefill
void flash_attention_prefill_forward(
    void *output,                  // [num_tokens, num_heads, head_dim]
    const void *query,             // [num_tokens, num_heads, head_dim]
    const void *key,               // [num_tokens, num_kv_heads, head_dim]
    const void *value,             // [num_tokens, num_kv_heads, head_dim]
    const int *positions,          // [num_tokens] - positions for each token
    int num_tokens,                // Number of tokens
    const FlashAttentionConfig &config,
    cudaStream_t stream);

// FP16 version
void flash_attention_prefill_forward_fp16(
    half *output,                  // [num_tokens, num_heads, head_dim]
    const half *query,             // [num_tokens, num_heads, head_dim]
    const half *key,               // [num_tokens, num_kv_heads, head_dim]
    const half *value,             // [num_tokens, num_kv_heads, head_dim]
    const int *positions,          // [num_tokens] - positions for each token
    int num_tokens,                // Number of tokens
    const FlashAttentionConfig &config,
    cudaStream_t stream);

} // namespace mini_vllm