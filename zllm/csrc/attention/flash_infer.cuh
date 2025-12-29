// =============================================================================
// flash_infer.cuh - FlashInfer (Decode) Attention Header
// =============================================================================

#pragma once

#include "../include/common.cuh"
#include <cuda_fp16.h>

namespace mini_vllm {

// =============================================================================
// FlashInfer Configuration
// =============================================================================

struct FlashInferConfig {
  int num_heads;           // Number of attention heads
  int num_kv_heads;        // Number of KV heads (for GQA)
  int head_dim;            // Dimension of each head
  float softmax_scale;     // Softmax scaling factor (1/sqrt(head_dim))
  int dtype_size;          // Size of data type (2 for FP16, 4 for FP32)
};

// =============================================================================
// FlashInfer Functions
// =============================================================================

// Forward pass for flash infer (decode)
void flash_infer_forward(
    void *output,                  // [num_tokens, num_heads, head_dim]
    const void *query,             // [num_tokens, num_heads, head_dim]
    const void *key_cache,         // [num_layers, num_blocks, num_kv_heads, block_size, head_dim]
    const void *value_cache,       // [num_layers, num_blocks, num_kv_heads, block_size, head_dim]
    const int *block_table,        // [num_sequences, max_blocks] - block IDs for each sequence
    const int *block_offsets,      // [num_tokens] - offset within block for each token
    const int *seq_lengths,        // [num_sequences] - length of each sequence
    int num_tokens,                // Number of tokens to process
    int num_sequences,             // Number of sequences
    const FlashInferConfig &config,
    cudaStream_t stream);

// FP16 version
void flash_infer_forward_fp16(
    half *output,                  // [num_tokens, num_heads, head_dim]
    const half *query,             // [num_tokens, num_heads, head_dim]
    const half *key_cache,         // [num_layers, num_blocks, num_kv_heads, block_size, head_dim]
    const half *value_cache,       // [num_layers, num_blocks, num_kv_heads, block_size, head_dim]
    const int *block_table,        // [num_sequences, max_blocks] - block IDs for each token
    const int *block_offsets,      // [num_tokens] - offset within block for each token
    const int *seq_lengths,        // [num_sequences] - length of each sequence
    int num_tokens,                // Number of tokens to process
    int num_sequences,             // Number of sequences
    const FlashInferConfig &config,
    cudaStream_t stream);

} // namespace mini_vllm