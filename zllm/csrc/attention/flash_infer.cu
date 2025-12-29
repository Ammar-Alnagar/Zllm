// =============================================================================
// flash_infer.cu - FlashInfer (Decode) Attention Implementation
// =============================================================================

#include "flash_infer.cuh"
#include "../include/common.cuh"
#include <cuda_fp16.h>
#include <float.h>

namespace mini_vllm {

// =============================================================================
// FlashInfer Kernel (FP32)
// =============================================================================

template <int BLOCK_SIZE, int HEAD_DIM, int BLOCK_TOKENS>
__global__ void flash_infer_kernel(
    float *__restrict__ output,           // [num_tokens, num_heads, head_dim]
    const float *__restrict__ query,      // [num_tokens, num_heads, head_dim]
    const float *__restrict__ key_cache,  // [num_blocks, num_kv_heads, block_size, head_dim]
    const float *__restrict__ value_cache,// [num_blocks, num_kv_heads, block_size, head_dim]
    const int *__restrict__ block_table,  // [num_sequences, max_blocks]
    const int *__restrict__ block_offsets,// [num_tokens]
    const int *__restrict__ seq_lengths,  // [num_sequences]
    int num_tokens, int num_sequences, int num_heads, int num_kv_heads,
    float softmax_scale) {
  
  // Block and thread indices
  const int token_idx = blockIdx.y;      // Current token
  const int head_idx = blockIdx.x;      // Current head
  const int tid = threadIdx.x;          // Thread within block
  
  // Determine which sequence this token belongs to
  // For simplicity, assume tokens are ordered by sequence
  int seq_idx = 0;
  int seq_start = 0;
  for (int s = 0; s < num_sequences; s++) {
    if (token_idx < seq_start + seq_lengths[s]) {
      seq_idx = s;
      break;
    }
    seq_start += seq_lengths[s];
  }
  
  // Get block information for this token
  int block_offset = block_offsets[token_idx];
  int block_idx = block_table[seq_idx * 100 + block_offset / BLOCK_TOKENS]; // Simplified
  int intra_block_offset = block_offset % BLOCK_TOKENS;
  
  // Shared memory for Q, K, V tiles
  __shared__ float q_tile[BLOCK_SIZE][HEAD_DIM];
  __shared__ float k_tile[BLOCK_SIZE][HEAD_DIM];
  __shared__ float v_tile[BLOCK_SIZE][HEAD_DIM];
  
  // Output accumulator and softmax statistics
  float o_acc[HEAD_DIM] = {0.0f};
  float max_val = -FLT_MAX;
  float sum_exp = 0.0f;
  
  // Load current query into shared memory
  if (tid < HEAD_DIM) {
    int q_idx = token_idx * num_heads * HEAD_DIM + head_idx * HEAD_DIM + tid;
    q_tile[0][tid] = query[q_idx];
  }
  __syncthreads();
  
  // Process all blocks for this sequence
  for (int b = 0; b <= block_offset / BLOCK_TOKENS; b++) {
    int current_block_idx = block_table[seq_idx * 100 + b];
    
    // Process each token in this block
    for (int t = 0; t < BLOCK_TOKENS; t++) {
      int pos_in_block = t;
      
      // Skip future positions (causal)
      if (b == block_offset / BLOCK_TOKENS && pos_in_block > intra_block_offset) {
        continue;
      }
      
      // Load K and V from cache
      if (tid < HEAD_DIM) {
        int k_idx = current_block_idx * num_kv_heads * BLOCK_TOKENS * HEAD_DIM +
                   head_idx * BLOCK_TOKENS * HEAD_DIM +
                   pos_in_block * HEAD_DIM + tid;
        int v_idx = current_block_idx * num_kv_heads * BLOCK_TOKENS * HEAD_DIM +
                   head_idx * BLOCK_TOKENS * HEAD_DIM +
                   pos_in_block * HEAD_DIM + tid;
        k_tile[0][tid] = key_cache[k_idx];
        v_tile[0][tid] = value_cache[v_idx];
      }
      __syncthreads();
      
      // Compute attention score
      float score = 0.0f;
      for (int d = 0; d < HEAD_DIM; d++) {
        score += q_tile[0][d] * k_tile[0][d];
      }
      
      // Online softmax
      score *= softmax_scale;
      if (score > max_val) {
        float diff = max_val - score;
        sum_exp = sum_exp * expf(diff) + 1.0f;
        max_val = score;
      } else {
        sum_exp += expf(score - max_val);
      }
      
      // Accumulate output
      float attn_weight = expf(score - max_val) / sum_exp;
      for (int d = 0; d < HEAD_DIM; d++) {
        o_acc[d] += attn_weight * v_tile[0][d];
      }
      
      __syncthreads();
    }
  }
  
  // Store output
  if (tid < HEAD_DIM) {
    int out_idx = token_idx * num_heads * HEAD_DIM + head_idx * HEAD_DIM + tid;
    output[out_idx] = o_acc[tid];
  }
}

// =============================================================================
// FlashInfer Wrapper (FP32)
// =============================================================================

void flash_infer_forward(
    void *output, const void *query, const void *key_cache, const void *value_cache,
    const int *block_table, const int *block_offsets, const int *seq_lengths,
    int num_tokens, int num_sequences, const FlashInferConfig &config,
    cudaStream_t stream) {
  
  // Cast to float pointers
  float *out = static_cast<float *>(output);
  const float *q = static_cast<const float *>(query);
  const float *k_cache = static_cast<const float *>(key_cache);
  const float *v_cache = static_cast<const float *>(value_cache);
  
  // Determine block size based on head dimension
  constexpr int BLOCK_SIZE = 256;
  constexpr int BLOCK_TOKENS = 16; // 16 tokens per block
  int head_dim = config.head_dim;
  
  // Launch kernel
  dim3 grid(config.num_heads, num_tokens);
  dim3 block(BLOCK_SIZE);
  
  // Use template based on head dimension
  switch (head_dim) {
    case 64:
      flash_infer_kernel<BLOCK_SIZE, 64, BLOCK_TOKENS>
          <<<grid, block, 0, stream>>>(out, q, k_cache, v_cache, block_table,
                                       block_offsets, seq_lengths, num_tokens,
                                       num_sequences, config.num_heads,
                                       config.num_kv_heads, config.softmax_scale);
      break;
    case 128:
      flash_infer_kernel<BLOCK_SIZE, 128, BLOCK_TOKENS>
          <<<grid, block, 0, stream>>>(out, q, k_cache, v_cache, block_table,
                                       block_offsets, seq_lengths, num_tokens,
                                       num_sequences, config.num_heads,
                                       config.num_kv_heads, config.softmax_scale);
      break;
    case 256:
      flash_infer_kernel<BLOCK_SIZE, 256, BLOCK_TOKENS>
          <<<grid, block, 0, stream>>>(out, q, k_cache, v_cache, block_table,
                                       block_offsets, seq_lengths, num_tokens,
                                       num_sequences, config.num_heads,
                                       config.num_kv_heads, config.softmax_scale);
      break;
    default:
      printf("Unsupported head dimension for FlashInfer: %d\n", head_dim);
      break;
  }
  
  CUDA_CHECK_LAST();
}

// =============================================================================
// FlashInfer Kernel (FP16)
// =============================================================================

template <int BLOCK_SIZE, int HEAD_DIM, int BLOCK_TOKENS>
__global__ void flash_infer_fp16_kernel(
    half *__restrict__ output,             // [num_tokens, num_heads, head_dim]
    const half *__restrict__ query,        // [num_tokens, num_heads, head_dim]
    const half *__restrict__ key_cache,    // [num_blocks, num_kv_heads, block_size, head_dim]
    const half *__restrict__ value_cache,  // [num_blocks, num_kv_heads, block_size, head_dim]
    const int *__restrict__ block_table,   // [num_sequences, max_blocks]
    const int *__restrict__ block_offsets, // [num_tokens]
    const int *__restrict__ seq_lengths,   // [num_sequences]
    int num_tokens, int num_sequences, int num_heads, int num_kv_heads,
    float softmax_scale) {
  
  // Block and thread indices
  const int token_idx = blockIdx.y;      // Current token
  const int head_idx = blockIdx.x;      // Current head
  const int tid = threadIdx.x;          // Thread within block
  
  // Determine which sequence this token belongs to
  int seq_idx = 0;
  int seq_start = 0;
  for (int s = 0; s < num_sequences; s++) {
    if (token_idx < seq_start + seq_lengths[s]) {
      seq_idx = s;
      break;
    }
    seq_start += seq_lengths[s];
  }
  
  // Get block information for this token
  int block_offset = block_offsets[token_idx];
  int block_idx = block_table[seq_idx * 100 + block_offset / BLOCK_TOKENS]; // Simplified
  int intra_block_offset = block_offset % BLOCK_TOKENS;
  
  // Shared memory for Q, K, V tiles (use float for computation)
  __shared__ float q_tile[BLOCK_SIZE][HEAD_DIM];
  __shared__ float k_tile[BLOCK_SIZE][HEAD_DIM];
  __shared__ float v_tile[BLOCK_SIZE][HEAD_DIM];
  
  // Output accumulator and softmax statistics
  float o_acc[HEAD_DIM] = {0.0f};
  float max_val = -FLT_MAX;
  float sum_exp = 0.0f;
  
  // Load current query into shared memory (convert from FP16)
  if (tid < HEAD_DIM) {
    int q_idx = token_idx * num_heads * HEAD_DIM + head_idx * HEAD_DIM + tid;
    q_tile[0][tid] = __half2float(query[q_idx]);
  }
  __syncthreads();
  
  // Process all blocks for this sequence
  for (int b = 0; b <= block_offset / BLOCK_TOKENS; b++) {
    int current_block_idx = block_table[seq_idx * 100 + b];
    
    // Process each token in this block
    for (int t = 0; t < BLOCK_TOKENS; t++) {
      int pos_in_block = t;
      
      // Skip future positions (causal)
      if (b == block_offset / BLOCK_TOKENS && pos_in_block > intra_block_offset) {
        continue;
      }
      
      // Load K and V from cache (convert from FP16)
      if (tid < HEAD_DIM) {
        int k_idx = current_block_idx * num_kv_heads * BLOCK_TOKENS * HEAD_DIM +
                   head_idx * BLOCK_TOKENS * HEAD_DIM +
                   pos_in_block * HEAD_DIM + tid;
        int v_idx = current_block_idx * num_kv_heads * BLOCK_TOKENS * HEAD_DIM +
                   head_idx * BLOCK_TOKENS * HEAD_DIM +
                   pos_in_block * HEAD_DIM + tid;
        k_tile[0][tid] = __half2float(key_cache[k_idx]);
        v_tile[0][tid] = __half2float(value_cache[v_idx]);
      }
      __syncthreads();
      
      // Compute attention score
      float score = 0.0f;
      for (int d = 0; d < HEAD_DIM; d++) {
        score += q_tile[0][d] * k_tile[0][d];
      }
      
      // Online softmax
      score *= softmax_scale;
      if (score > max_val) {
        float diff = max_val - score;
        sum_exp = sum_exp * expf(diff) + 1.0f;
        max_val = score;
      } else {
        sum_exp += expf(score - max_val);
      }
      
      // Accumulate output
      float attn_weight = expf(score - max_val) / sum_exp;
      for (int d = 0; d < HEAD_DIM; d++) {
        o_acc[d] += attn_weight * v_tile[0][d];
      }
      
      __syncthreads();
    }
  }
  
  // Store output (convert back to FP16)
  if (tid < HEAD_DIM) {
    int out_idx = token_idx * num_heads * HEAD_DIM + head_idx * HEAD_DIM + tid;
    output[out_idx] = __float2half(o_acc[tid]);
  }
}

// =============================================================================
// FlashInfer Wrapper (FP16)
// =============================================================================

void flash_infer_forward_fp16(
    half *output, const half *query, const half *key_cache, const half *value_cache,
    const int *block_table, const int *block_offsets, const int *seq_lengths,
    int num_tokens, int num_sequences, const FlashInferConfig &config,
    cudaStream_t stream) {
  
  // Determine block size based on head dimension
  constexpr int BLOCK_SIZE = 256;
  constexpr int BLOCK_TOKENS = 16; // 16 tokens per block
  int head_dim = config.head_dim;
  
  // Launch kernel
  dim3 grid(config.num_heads, num_tokens);
  dim3 block(BLOCK_SIZE);
  
  // Use template based on head dimension
  switch (head_dim) {
    case 64:
      flash_infer_fp16_kernel<BLOCK_SIZE, 64, BLOCK_TOKENS>
          <<<grid, block, 0, stream>>>(output, query, key_cache, value_cache, block_table,
                                       block_offsets, seq_lengths, num_tokens,
                                       num_sequences, config.num_heads,
                                       config.num_kv_heads, config.softmax_scale);
      break;
    case 128:
      flash_infer_fp16_kernel<BLOCK_SIZE, 128, BLOCK_TOKENS>
          <<<grid, block, 0, stream>>>(output, query, key_cache, value_cache, block_table,
                                       block_offsets, seq_lengths, num_tokens,
                                       num_sequences, config.num_heads,
                                       config.num_kv_heads, config.softmax_scale);
      break;
    case 256:
      flash_infer_fp16_kernel<BLOCK_SIZE, 256, BLOCK_TOKENS>
          <<<grid, block, 0, stream>>>(output, query, key_cache, value_cache, block_table,
                                       block_offsets, seq_lengths, num_tokens,
                                       num_sequences, config.num_heads,
                                       config.num_kv_heads, config.softmax_scale);
      break;
    default:
      printf("Unsupported head dimension for FlashInfer FP16: %d\n", head_dim);
      break;
  }
  
  CUDA_CHECK_LAST();
}

} // namespace mini_vllm