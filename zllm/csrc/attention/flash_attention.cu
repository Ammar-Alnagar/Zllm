// =============================================================================
// flash_attention.cu - Flash Attention Prefill Implementation
// =============================================================================

#include "flash_attention.cuh"
#include "../include/common.cuh"
#include <cuda_fp16.h>
#include <float.h>

namespace mini_vllm {

// =============================================================================
// Online Softmax Helper Functions
// =============================================================================

// Compute online softmax for a single row
__device__ __forceinline__ void online_softmax(
    float &max_val, float &sum_exp, float new_val, float scale) {
  // Scale the new value
  new_val *= scale;
  
  // Update max
  if (new_val > max_val) {
    float diff = max_val - new_val;
    sum_exp = sum_exp * expf(diff) + 1.0f;
    max_val = new_val;
  } else {
    sum_exp += expf(new_val - max_val);
  }
}

// =============================================================================
// Flash Attention Prefill Kernel (FP32)
// =============================================================================

template <int BLOCK_SIZE, int HEAD_DIM>
__global__ void flash_attention_prefill_kernel(
    float *__restrict__ output,           // [num_tokens, num_heads, head_dim]
    const float *__restrict__ query,      // [num_tokens, num_heads, head_dim]
    const float *__restrict__ key,        // [num_tokens, num_kv_heads, head_dim]
    const float *__restrict__ value,      // [num_tokens, num_kv_heads, head_dim]
    const int *__restrict__ positions,    // [num_tokens]
    int num_tokens, int num_heads, int num_kv_heads,
    float softmax_scale, bool is_causal) {
  
  // Block and thread indices
  const int token_idx = blockIdx.y;      // Current token
  const int head_idx = blockIdx.x;      // Current head
  const int tid = threadIdx.x;          // Thread within block
  
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
  
  // Process each key/value token
  for (int k_token = 0; k_token < num_tokens; k_token++) {
    
    // Check causal masking
    if (is_causal && positions[k_token] > positions[token_idx]) {
      continue;  // Skip future tokens
    }
    
    // Load K and V tiles
    if (tid < HEAD_DIM) {
      int k_idx = k_token * num_kv_heads * HEAD_DIM + head_idx * HEAD_DIM + tid;
      int v_idx = k_token * num_kv_heads * HEAD_DIM + head_idx * HEAD_DIM + tid;
      k_tile[0][tid] = key[k_idx];
      v_tile[0][tid] = value[v_idx];
    }
    __syncthreads();
    
    // Compute attention score
    float score = 0.0f;
    for (int d = 0; d < HEAD_DIM; d++) {
      score += q_tile[0][d] * k_tile[0][d];
    }
    
    // Online softmax
    online_softmax(max_val, sum_exp, score, softmax_scale);
    
    // Accumulate output
    float attn_weight = expf(score - max_val) / sum_exp;
    for (int d = 0; d < HEAD_DIM; d++) {
      o_acc[d] += attn_weight * v_tile[0][d];
    }
    
    __syncthreads();
  }
  
  // Store output
  if (tid < HEAD_DIM) {
    int out_idx = token_idx * num_heads * HEAD_DIM + head_idx * HEAD_DIM + tid;
    output[out_idx] = o_acc[tid];
  }
}

// =============================================================================
// Flash Attention Prefill Wrapper (FP32)
// =============================================================================

void flash_attention_prefill_forward(
    void *output, const void *query, const void *key, const void *value,
    const int *positions, int num_tokens, const FlashAttentionConfig &config,
    cudaStream_t stream) {
  
  // Cast to float pointers
  float *out = static_cast<float *>(output);
  const float *q = static_cast<const float *>(query);
  const float *k = static_cast<const float *>(key);
  const float *v = static_cast<const float *>(value);
  
  // Determine block size based on head dimension
  constexpr int BLOCK_SIZE = 256;
  int head_dim = config.head_dim;
  
  // Launch kernel
  dim3 grid(config.num_heads, num_tokens);
  dim3 block(BLOCK_SIZE);
  
  // Use template based on head dimension
  switch (head_dim) {
    case 64:
      flash_attention_prefill_kernel<BLOCK_SIZE, 64>
          <<<grid, block, 0, stream>>>(out, q, k, v, positions, num_tokens,
                                       config.num_heads, config.num_kv_heads,
                                       config.softmax_scale, config.is_causal);
      break;
    case 128:
      flash_attention_prefill_kernel<BLOCK_SIZE, 128>
          <<<grid, block, 0, stream>>>(out, q, k, v, positions, num_tokens,
                                       config.num_heads, config.num_kv_heads,
                                       config.softmax_scale, config.is_causal);
      break;
    case 256:
      flash_attention_prefill_kernel<BLOCK_SIZE, 256>
          <<<grid, block, 0, stream>>>(out, q, k, v, positions, num_tokens,
                                       config.num_heads, config.num_kv_heads,
                                       config.softmax_scale, config.is_causal);
      break;
    default:
      printf("Unsupported head dimension: %d\n", head_dim);
      break;
  }
  
  CUDA_CHECK_LAST();
}

// =============================================================================
// Flash Attention Prefill Kernel (FP16)
// =============================================================================

template <int BLOCK_SIZE, int HEAD_DIM>
__global__ void flash_attention_prefill_fp16_kernel(
    half *__restrict__ output,             // [num_tokens, num_heads, head_dim]
    const half *__restrict__ query,        // [num_tokens, num_heads, head_dim]
    const half *__restrict__ key,          // [num_tokens, num_kv_heads, head_dim]
    const half *__restrict__ value,        // [num_tokens, num_kv_heads, head_dim]
    const int *__restrict__ positions,    // [num_tokens]
    int num_tokens, int num_heads, int num_kv_heads,
    float softmax_scale, bool is_causal) {
  
  // Block and thread indices
  const int token_idx = blockIdx.y;      // Current token
  const int head_idx = blockIdx.x;      // Current head
  const int tid = threadIdx.x;          // Thread within block
  
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
  
  // Process each key/value token
  for (int k_token = 0; k_token < num_tokens; k_token++) {
    
    // Check causal masking
    if (is_causal && positions[k_token] > positions[token_idx]) {
      continue;  // Skip future tokens
    }
    
    // Load K and V tiles (convert from FP16)
    if (tid < HEAD_DIM) {
      int k_idx = k_token * num_kv_heads * HEAD_DIM + head_idx * HEAD_DIM + tid;
      int v_idx = k_token * num_kv_heads * HEAD_DIM + head_idx * HEAD_DIM + tid;
      k_tile[0][tid] = __half2float(key[k_idx]);
      v_tile[0][tid] = __half2float(value[v_idx]);
    }
    __syncthreads();
    
    // Compute attention score
    float score = 0.0f;
    for (int d = 0; d < HEAD_DIM; d++) {
      score += q_tile[0][d] * k_tile[0][d];
    }
    
    // Online softmax
    online_softmax(max_val, sum_exp, score, softmax_scale);
    
    // Accumulate output
    float attn_weight = expf(score - max_val) / sum_exp;
    for (int d = 0; d < HEAD_DIM; d++) {
      o_acc[d] += attn_weight * v_tile[0][d];
    }
    
    __syncthreads();
  }
  
  // Store output (convert back to FP16)
  if (tid < HEAD_DIM) {
    int out_idx = token_idx * num_heads * HEAD_DIM + head_idx * HEAD_DIM + tid;
    output[out_idx] = __float2half(o_acc[tid]);
  }
}

// =============================================================================
// Flash Attention Prefill Wrapper (FP16)
// =============================================================================

void flash_attention_prefill_forward_fp16(
    half *output, const half *query, const half *key, const half *value,
    const int *positions, int num_tokens, const FlashAttentionConfig &config,
    cudaStream_t stream) {
  
  // Determine block size based on head dimension
  constexpr int BLOCK_SIZE = 256;
  int head_dim = config.head_dim;
  
  // Launch kernel
  dim3 grid(config.num_heads, num_tokens);
  dim3 block(BLOCK_SIZE);
  
  // Use template based on head dimension
  switch (head_dim) {
    case 64:
      flash_attention_prefill_fp16_kernel<BLOCK_SIZE, 64>
          <<<grid, block, 0, stream>>>(output, query, key, value, positions, num_tokens,
                                       config.num_heads, config.num_kv_heads,
                                       config.softmax_scale, config.is_causal);
      break;
    case 128:
      flash_attention_prefill_fp16_kernel<BLOCK_SIZE, 128>
          <<<grid, block, 0, stream>>>(output, query, key, value, positions, num_tokens,
                                       config.num_heads, config.num_kv_heads,
                                       config.softmax_scale, config.is_causal);
      break;
    case 256:
      flash_attention_prefill_fp16_kernel<BLOCK_SIZE, 256>
          <<<grid, block, 0, stream>>>(output, query, key, value, positions, num_tokens,
                                       config.num_heads, config.num_kv_heads,
                                       config.softmax_scale, config.is_causal);
      break;
    default:
      printf("Unsupported head dimension for FP16: %d\n", head_dim);
      break;
  }
  
  CUDA_CHECK_LAST();
}

} // namespace mini_vllm