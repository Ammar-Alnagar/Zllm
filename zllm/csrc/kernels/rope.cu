// =============================================================================
// rope.cu - Rotary Position Embedding Implementation
// =============================================================================

#include "../include/common.cuh"
#include "rope.cuh"
#include <cmath>

namespace mini_vllm {

// =============================================================================
// Frequency Table Initialization
// =============================================================================

/**
 * Kernel to compute cos/sin tables
 *
 * Each thread computes one (position, dim_pair) entry
 */
__global__ void
rope_init_kernel(float *__restrict__ cos_table, // [max_seq_len, head_dim/2]
                 float *__restrict__ sin_table, // [max_seq_len, head_dim/2]
                 int max_seq_len, int half_head_dim, float theta_base) {
  // Position index
  int pos = blockIdx.x;
  // Dimension pair index
  int dim = threadIdx.x;

  if (pos < max_seq_len && dim < half_head_dim) {
    // Compute frequency: theta_i = 1 / (base^(2i/d))
    // = base^(-2i/d)
    float freq = powf(theta_base, -2.0f * dim / (2.0f * half_head_dim));

    // Compute angle for this position
    float angle = pos * freq;

    // Store cos and sin
    int idx = pos * half_head_dim + dim;
    cos_table[idx] = cosf(angle);
    sin_table[idx] = sinf(angle);
  }
}

void rope_init_tables(float *cos_table, float *sin_table, int max_seq_len,
                      int head_dim, float theta_base, cudaStream_t stream) {
  int half_head_dim = head_dim / 2;

  // Launch one block per position, threads per dimension pair
  dim3 grid(max_seq_len);
  dim3 block(half_head_dim);

  rope_init_kernel<<<grid, block, 0, stream>>>(
      cos_table, sin_table, max_seq_len, half_head_dim, theta_base);

  CUDA_CHECK_LAST();
}

// =============================================================================
// RoPE Forward Pass
// =============================================================================

/**
 * rope_kernel - Apply rotary embeddings to Q and K
 *
 * Memory layout:
 * Q: [num_tokens, num_heads, head_dim]
 * K: [num_tokens, num_kv_heads, head_dim]
 *
 * For GQA: num_heads > num_kv_heads (Q heads grouped to share KV)
 *
 * Each block handles one (token, head) pair
 * Each thread handles one dimension pair
 */
__global__ void
rope_kernel(float *__restrict__ query, // [num_tokens, num_heads, head_dim]
            float *__restrict__ key,   // [num_tokens, num_kv_heads, head_dim]
            const float *__restrict__ cos_table, // [max_seq_len, head_dim/2]
            const float *__restrict__ sin_table, // [max_seq_len, head_dim/2]
            const int *__restrict__ positions,   // [num_tokens]
            int num_tokens, int num_heads, int num_kv_heads, int head_dim) {
  // Block indices
  const int token_idx = blockIdx.y;
  const int head_idx = blockIdx.x;

  // Thread index = dimension pair index
  const int dim_pair = threadIdx.x;
  const int half_head_dim = head_dim / 2;

  if (dim_pair >= half_head_dim)
    return;

  // Get position for this token
  const int pos = positions[token_idx];

  // Load cos/sin for this position and dimension
  const int table_idx = pos * half_head_dim + dim_pair;
  const float cos_val = cos_table[table_idx];
  const float sin_val = sin_table[table_idx];

  // =========================================================================
  // Apply RoPE to Query
  // =========================================================================
  if (head_idx < num_heads) {
    // Calculate indices for the dimension pair in Q
    // Q layout: [num_tokens, num_heads, head_dim]
    int q_base = token_idx * num_heads * head_dim + head_idx * head_dim;
    int q_idx_even = q_base + 2 * dim_pair;
    int q_idx_odd = q_base + 2 * dim_pair + 1;

    // Load Q values
    float q_even = query[q_idx_even];
    float q_odd = query[q_idx_odd];

    // Apply rotation:
    // q'_even = q_even * cos - q_odd * sin
    // q'_odd  = q_even * sin + q_odd * cos
    float q_rotated_even = q_even * cos_val - q_odd * sin_val;
    float q_rotated_odd = q_even * sin_val + q_odd * cos_val;

    // Store rotated values
    query[q_idx_even] = q_rotated_even;
    query[q_idx_odd] = q_rotated_odd;
  }

  // =========================================================================
  // Apply RoPE to Key
  // =========================================================================
  // Only apply to KV heads (fewer than Q heads in GQA)
  if (head_idx < num_kv_heads) {
    // K layout: [num_tokens, num_kv_heads, head_dim]
    int k_base = token_idx * num_kv_heads * head_dim + head_idx * head_dim;
    int k_idx_even = k_base + 2 * dim_pair;
    int k_idx_odd = k_base + 2 * dim_pair + 1;

    // Load K values
    float k_even = key[k_idx_even];
    float k_odd = key[k_idx_odd];

    // Apply rotation
    float k_rotated_even = k_even * cos_val - k_odd * sin_val;
    float k_rotated_odd = k_even * sin_val + k_odd * cos_val;

    // Store rotated values
    key[k_idx_even] = k_rotated_even;
    key[k_idx_odd] = k_rotated_odd;
  }
}

void rope_forward(float *query, float *key, const float *cos_table,
                  const float *sin_table, const int *positions, int num_tokens,
                  int num_heads, int num_kv_heads, int head_dim,
                  cudaStream_t stream) {
  int half_head_dim = head_dim / 2;

  // Grid: (max(num_heads, num_kv_heads), num_tokens)
  // Block: half_head_dim threads
  int max_heads = std::max(num_heads, num_kv_heads);

  dim3 grid(max_heads, num_tokens);
  dim3 block(half_head_dim);

  rope_kernel<<<grid, block, 0, stream>>>(query, key, cos_table, sin_table,
                                          positions, num_tokens, num_heads,
                                          num_kv_heads, head_dim);

  CUDA_CHECK_LAST();
}

// =============================================================================
// Optimized RoPE with Vectorization
// =============================================================================

/**
 * Optimized RoPE kernel with:
 * 1. Vectorized loads/stores (float2)
 * 2. Better memory coalescing
 * 3. Fused Q and K processing
 */
__global__ void rope_optimized_kernel(float *__restrict__ query,
                                      float *__restrict__ key,
                                      const float *__restrict__ cos_table,
                                      const float *__restrict__ sin_table,
                                      const int *__restrict__ positions,
                                      int num_tokens, int num_heads,
                                      int num_kv_heads, int head_dim) {
  const int token_idx = blockIdx.y;
  const int head_idx = blockIdx.x;
  const int dim_pair = threadIdx.x;
  const int half_head_dim = head_dim / 2;

  if (dim_pair >= half_head_dim)
    return;

  // Get position and load cos/sin (these are cached in L1)
  const int pos = positions[token_idx];
  const int table_idx = pos * half_head_dim + dim_pair;

  // Use float2 for cos/sin pair
  const float cos_val = cos_table[table_idx];
  const float sin_val = sin_table[table_idx];

  // Process Query
  if (head_idx < num_heads) {
    int q_base = token_idx * num_heads * head_dim + head_idx * head_dim;

    // Load as float2 (8 bytes, coalesced)
    float2 *q_ptr = reinterpret_cast<float2 *>(&query[q_base + 2 * dim_pair]);
    float2 q = *q_ptr;

    // Rotate
    float2 q_rot;
    q_rot.x = q.x * cos_val - q.y * sin_val;
    q_rot.y = q.x * sin_val + q.y * cos_val;

    // Store
    *q_ptr = q_rot;
  }

  // Process Key
  if (head_idx < num_kv_heads) {
    int k_base = token_idx * num_kv_heads * head_dim + head_idx * head_dim;

    float2 *k_ptr = reinterpret_cast<float2 *>(&key[k_base + 2 * dim_pair]);
    float2 k = *k_ptr;

    float2 k_rot;
    k_rot.x = k.x * cos_val - k.y * sin_val;
    k_rot.y = k.x * sin_val + k.y * cos_val;

    *k_ptr = k_rot;
  }
}

// =============================================================================
// FP16 Implementation
// =============================================================================

__global__ void rope_fp16_kernel(half *__restrict__ query,
                                 half *__restrict__ key,
                                 const float *__restrict__ cos_table,
                                 const float *__restrict__ sin_table,
                                 const int *__restrict__ positions,
                                 int num_tokens, int num_heads,
                                 int num_kv_heads, int head_dim) {
  const int token_idx = blockIdx.y;
  const int head_idx = blockIdx.x;
  const int dim_pair = threadIdx.x;
  const int half_head_dim = head_dim / 2;

  if (dim_pair >= half_head_dim)
    return;

  const int pos = positions[token_idx];
  const int table_idx = pos * half_head_dim + dim_pair;

  // Keep cos/sin in FP32 for accuracy
  const float cos_val = cos_table[table_idx];
  const float sin_val = sin_table[table_idx];

  // Process Query
  if (head_idx < num_heads) {
    int q_base = token_idx * num_heads * head_dim + head_idx * head_dim;
    int q_idx = q_base + 2 * dim_pair;

    // Load half2 (both elements of the pair)
    half2 *q_ptr = reinterpret_cast<half2 *>(&query[q_idx]);
    half2 q_half = *q_ptr;

    // Convert to float for computation
    float2 q = __half22float2(q_half);

    // Rotate
    float2 q_rot;
    q_rot.x = q.x * cos_val - q.y * sin_val;
    q_rot.y = q.x * sin_val + q.y * cos_val;

    // Convert back and store
    *q_ptr = __float22half2_rn(q_rot);
  }

  // Process Key
  if (head_idx < num_kv_heads) {
    int k_base = token_idx * num_kv_heads * head_dim + head_idx * head_dim;
    int k_idx = k_base + 2 * dim_pair;

    half2 *k_ptr = reinterpret_cast<half2 *>(&key[k_idx]);
    half2 k_half = *k_ptr;

    float2 k = __half22float2(k_half);

    float2 k_rot;
    k_rot.x = k.x * cos_val - k.y * sin_val;
    k_rot.y = k.x * sin_val + k.y * cos_val;

    *k_ptr = __float22half2_rn(k_rot);
  }
}

void rope_forward_fp16(half *query, half *key, const float *cos_table,
                       const float *sin_table, const int *positions,
                       int num_tokens, int num_heads, int num_kv_heads,
                       int head_dim, cudaStream_t stream) {
  int half_head_dim = head_dim / 2;
  int max_heads = max(num_heads, num_kv_heads);

  dim3 grid(max_heads, num_tokens);
  dim3 block(half_head_dim);

  rope_fp16_kernel<<<grid, block, 0, stream>>>(query, key, cos_table, sin_table,
                                               positions, num_tokens, num_heads,
                                               num_kv_heads, head_dim);

  CUDA_CHECK_LAST();
}

} // namespace mini_vllm
