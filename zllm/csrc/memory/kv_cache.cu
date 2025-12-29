// =============================================================================
// kv_cache.cu - KV Cache CUDA Operations Implementation
// =============================================================================

#include "kv_cache.cuh"

namespace mini_vllm {

// =============================================================================
// Copy Kernel
// =============================================================================

/**
 * kv_cache_copy_kernel - Copy KV values into paged cache
 *
 * Each thread handles one element.
 *
 * Slot mapping: For token t, slot_mapping[t] gives the linear slot index.
 * To convert slot → (block_id, offset):
 *   block_id = slot / block_size
 *   offset = slot % block_size
 */
__global__ void kv_cache_copy_kernel(
    float* __restrict__ kv_cache,    // [num_blocks, 2, block_size, num_kv_heads, head_dim]
    const float* __restrict__ keys,   // [num_tokens, num_kv_heads, head_dim]
    const float* __restrict__ values, // [num_tokens, num_kv_heads, head_dim]
    const int* __restrict__ slot_mapping, // [num_tokens]
    int num_tokens,
    int num_kv_heads,
    int head_dim,
    int block_size
) {
    // Each thread handles one (token, head, dim) element
    const int total_elements = num_tokens * num_kv_heads * head_dim;

    for (int idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < total_elements;
         idx += blockDim.x * gridDim.x) {

        // Decompose linear index
        int token_idx = idx / (num_kv_heads * head_dim);
        int remainder = idx % (num_kv_heads * head_dim);
        int head_idx = remainder / head_dim;
        int dim_idx = remainder % head_dim;

        // Get cache slot
        int slot = slot_mapping[token_idx];
        int block_id = slot / block_size;
        int block_offset = slot % block_size;

        // Source index
        int src_idx = token_idx * num_kv_heads * head_dim +
                      head_idx * head_dim + dim_idx;

        // Cache indices
        // Layout: [num_blocks, 2, block_size, num_kv_heads, head_dim]
        int kv_stride = block_size * num_kv_heads * head_dim;
        int block_stride = 2 * kv_stride;

        int k_idx = block_id * block_stride +
                    0 * kv_stride +  // K
                    block_offset * num_kv_heads * head_dim +
                    head_idx * head_dim + dim_idx;

        int v_idx = block_id * block_stride +
                    1 * kv_stride +  // V
                    block_offset * num_kv_heads * head_dim +
                    head_idx * head_dim + dim_idx;

        // Copy
        kv_cache[k_idx] = keys[src_idx];
        kv_cache[v_idx] = values[src_idx];
    }
}

void kv_cache_copy(
    float* kv_cache,
    const float* keys,
    const float* values,
    const int* slot_mapping,
    int num_tokens,
    int num_kv_heads,
    int head_dim,
    int block_size,
    cudaStream_t stream
) {
    const int total = num_tokens * num_kv_heads * head_dim;
    const int block_dim = 256;
    const int grid_dim = (total + block_dim - 1) / block_dim;

    kv_cache_copy_kernel<<<grid_dim, block_dim, 0, stream>>>(
        kv_cache, keys, values, slot_mapping,
        num_tokens, num_kv_heads, head_dim, block_size
    );

    CUDA_CHECK_LAST();
}

// =============================================================================
// Vectorized Copy (More Efficient)
// =============================================================================

/**
 * Optimized copy using float4 for 128-bit transactions
 */
template<int VEC_SIZE>
__global__ void kv_cache_copy_vectorized_kernel(
    float* __restrict__ kv_cache,
    const float* __restrict__ keys,
    const float* __restrict__ values,
    const int* __restrict__ slot_mapping,
    int num_tokens,
    int num_kv_heads,
    int head_dim,
    int block_size
) {
    // Assumes head_dim is divisible by VEC_SIZE (usually 4)
    const int vec_head_dim = head_dim / VEC_SIZE;
    const int total_vecs = num_tokens * num_kv_heads * vec_head_dim;

    for (int idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < total_vecs;
         idx += blockDim.x * gridDim.x) {

        int token_idx = idx / (num_kv_heads * vec_head_dim);
        int remainder = idx % (num_kv_heads * vec_head_dim);
        int head_idx = remainder / vec_head_dim;
        int vec_idx = remainder % vec_head_dim;

        int slot = slot_mapping[token_idx];
        int block_id = slot / block_size;
        int block_offset = slot % block_size;

        // Source (vectorized)
        int src_base = token_idx * num_kv_heads * head_dim +
                       head_idx * head_dim + vec_idx * VEC_SIZE;

        // Cache layout
        int kv_stride = block_size * num_kv_heads * head_dim;
        int block_stride = 2 * kv_stride;

        int dst_base_k = block_id * block_stride +
                         0 * kv_stride +
                         block_offset * num_kv_heads * head_dim +
                         head_idx * head_dim + vec_idx * VEC_SIZE;

        int dst_base_v = block_id * block_stride +
                         1 * kv_stride +
                         block_offset * num_kv_heads * head_dim +
                         head_idx * head_dim + vec_idx * VEC_SIZE;

        // Vectorized load and store
        if constexpr (VEC_SIZE == 4) {
            float4 k_vec = *reinterpret_cast<const float4*>(&keys[src_base]);
            float4 v_vec = *reinterpret_cast<const float4*>(&values[src_base]);

            *reinterpret_cast<float4*>(&kv_cache[dst_base_k]) = k_vec;
            *reinterpret_cast<float4*>(&kv_cache[dst_base_v]) = v_vec;
        }
    }
}

// =============================================================================
// Block Copy Kernel
// =============================================================================

__global__ void kv_cache_copy_blocks_kernel(
    float* __restrict__ dst_cache,
    const float* __restrict__ src_cache,
    const int* __restrict__ block_mapping,  // [num_pairs, 2] - (src, dst) pairs
    int num_pairs,
    int elements_per_block  // block_size * num_kv_heads * head_dim * 2
) {
    const int pair_idx = blockIdx.x;
    if (pair_idx >= num_pairs) return;

    const int src_block = block_mapping[pair_idx * 2];
    const int dst_block = block_mapping[pair_idx * 2 + 1];

    // Copy all elements in the block
    for (int i = threadIdx.x; i < elements_per_block; i += blockDim.x) {
        dst_cache[dst_block * elements_per_block + i] =
            src_cache[src_block * elements_per_block + i];
    }
}

void kv_cache_copy_blocks(
    float* dst_cache,
    const float* src_cache,
    const int* block_mapping,
    int num_blocks,
    int block_size,
    int num_kv_heads,
    int head_dim,
    cudaStream_t stream
) {
    int elements = 2 * block_size * num_kv_heads * head_dim;

    kv_cache_copy_blocks_kernel<<<num_blocks, 256, 0, stream>>>(
        dst_cache, src_cache, block_mapping, num_blocks, elements
    );

    CUDA_CHECK_LAST();
}

// =============================================================================
// FP16 Implementations
// =============================================================================

__global__ void kv_cache_copy_fp16_kernel(
    half* __restrict__ kv_cache,
    const half* __restrict__ keys,
    const half* __restrict__ values,
    const int* __restrict__ slot_mapping,
    int num_tokens,
    int num_kv_heads,
    int head_dim,
    int block_size
) {
    // Use half2 for better efficiency
    const int vec_head_dim = head_dim / 2;
    const int total_vecs = num_tokens * num_kv_heads * vec_head_dim;

    for (int idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < total_vecs;
         idx += blockDim.x * gridDim.x) {

        int token_idx = idx / (num_kv_heads * vec_head_dim);
        int remainder = idx % (num_kv_heads * vec_head_dim);
        int head_idx = remainder / vec_head_dim;
        int vec_idx = remainder % vec_head_dim;

        int slot = slot_mapping[token_idx];
        int block_id = slot / block_size;
        int block_offset = slot % block_size;

        int src_base = token_idx * num_kv_heads * head_dim +
                       head_idx * head_dim + vec_idx * 2;

        int kv_stride = block_size * num_kv_heads * head_dim;
        int block_stride = 2 * kv_stride;

        int dst_k = block_id * block_stride +
                    block_offset * num_kv_heads * head_dim +
                    head_idx * head_dim + vec_idx * 2;

        int dst_v = dst_k + kv_stride;

        half2 k_vec = *reinterpret_cast<const half2*>(&keys[src_base]);
        half2 v_vec = *reinterpret_cast<const half2*>(&values[src_base]);

        *reinterpret_cast<half2*>(&kv_cache[dst_k]) = k_vec;
        *reinterpret_cast<half2*>(&kv_cache[dst_v]) = v_vec;
    }
}

void kv_cache_copy_fp16(
    half* kv_cache,
    const half* keys,
    const half* values,
    const int* slot_mapping,
    int num_tokens,
    int num_kv_heads,
    int head_dim,
    int block_size,
    cudaStream_t stream
) {
    const int total = num_tokens * num_kv_heads * head_dim / 2;
    const int block_dim = 256;
    const int grid_dim = (total + block_dim - 1) / block_dim;

    kv_cache_copy_fp16_kernel<<<grid_dim, block_dim, 0, stream>>>(
        kv_cache, keys, values, slot_mapping,
        num_tokens, num_kv_heads, head_dim, block_size
    );

    CUDA_CHECK_LAST();
}

} // namespace mini_vllm