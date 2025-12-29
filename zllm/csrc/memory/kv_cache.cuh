// =============================================================================
// kv_cache.cuh - KV Cache CUDA Operations
// =============================================================================

#pragma once

#include "common.cuh"
#include <cuda_runtime.h>
#include <cuda_fp16.h>

namespace mini_vllm {

/**
 * Copy new KV values into cache blocks
 *
 * Used after computing K and V projections to store them in paged cache.
 *
 * @param kv_cache: KV cache [num_blocks, 2, block_size, num_kv_heads, head_dim]
 * @param keys: New K values [num_tokens, num_kv_heads, head_dim]
 * @param values: New V values [num_tokens, num_kv_heads, head_dim]
 * @param slot_mapping: Maps token index to cache slot [num_tokens]
 * @param num_tokens: Number of tokens to copy
 * @param num_kv_heads: Number of KV heads
 * @param head_dim: Dimension per head
 * @param block_size: Tokens per block
 * @param stream: CUDA stream
 */
void kv_cache_copy(
    float* kv_cache,
    const float* keys,
    const float* values,
    const int* slot_mapping,
    int num_tokens,
    int num_kv_heads,
    int head_dim,
    int block_size,
    cudaStream_t stream = nullptr
);

// FP16 version
void kv_cache_copy_fp16(
    half* kv_cache,
    const half* keys,
    const half* values,
    const int* slot_mapping,
    int num_tokens,
    int num_kv_heads,
    int head_dim,
    int block_size,
    cudaStream_t stream = nullptr
);

/**
 * Copy KV cache blocks (for prefix sharing / fork)
 *
 * @param dst_cache: Destination cache
 * @param src_cache: Source cache
 * @param block_mapping: src_block -> dst_block mapping
 * @param num_blocks: Number of blocks to copy
 * @param block_size: Tokens per block
 * @param num_kv_heads: Number of KV heads
 * @param head_dim: Head dimension
 * @param stream: CUDA stream
 */
void kv_cache_copy_blocks(
    float* dst_cache,
    const float* src_cache,
    const int* block_mapping,
    int num_blocks,
    int block_size,
    int num_kv_heads,
    int head_dim,
    cudaStream_t stream = nullptr
);

/**
 * Reshape KV cache from one layout to another
 *
 * Used for converting between different kernel expectations.
 */
void kv_cache_reshape(
    float* dst,
    const float* src,
    int num_blocks,
    int block_size,
    int num_kv_heads,
    int head_dim,
    cudaStream_t stream = nullptr
);

} // namespace mini_vllm