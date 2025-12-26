// =============================================================================
// types.hpp - Type Definitions
// =============================================================================
// Common type definitions and data structures used throughout the project.
// =============================================================================

#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace mini_vllm {

// =============================================================================
// Basic Types
// =============================================================================

// Data types supported by the inference engine
enum class DataType {
  FP32, // 32-bit floating point
  FP16, // 16-bit floating point (half precision)
  BF16, // Brain floating point (16-bit)
  INT8, // 8-bit integer (for quantization)
  INT4  // 4-bit integer (for quantization)
};

// Get size in bytes for a data type
inline size_t dtype_size(DataType dtype) {
  switch (dtype) {
  case DataType::FP32:
    return 4;
  case DataType::FP16:
    return 2;
  case DataType::BF16:
    return 2;
  case DataType::INT8:
    return 1;
  case DataType::INT4:
    return 1; // Packed as 2 values per byte
  default:
    return 0;
  }
}

// =============================================================================
// Model Configuration
// =============================================================================

/**
 * ModelConfig - Configuration for a transformer model
 *
 * This matches the Qwen3 architecture with GQA support.
 */
struct ModelConfig {
  // Model dimensions
  int hidden_size = 4096;             // Hidden dimension (d_model)
  int intermediate_size = 11008;      // FFN intermediate dimension
  int num_hidden_layers = 32;         // Number of transformer layers
  int num_attention_heads = 32;       // Number of query heads
  int num_key_value_heads = 8;        // Number of KV heads (GQA)
  int head_dim = 128;                 // Dimension per head
  int vocab_size = 152064;            // Vocabulary size (~150K)
  int max_position_embeddings = 8192; // Maximum sequence length

  // RoPE configuration
  float rope_theta = 1000000.0f; // RoPE base frequency

  // Normalization
  float rms_norm_eps = 1e-6f; // RMSNorm epsilon

  // Data type
  DataType dtype = DataType::FP16;

  // Derived values (computed from above)
  int kv_head_ratio() const {
    return num_attention_heads / num_key_value_heads;
  }

  size_t kv_cache_size_per_token() const {
    // K + V for each layer and each KV head
    return 2 * num_hidden_layers * num_key_value_heads * head_dim *
           dtype_size(dtype);
  }
};

// =============================================================================
// KV Cache Types
// =============================================================================

// Block size for paged KV cache (tokens per block)
constexpr int KV_BLOCK_SIZE = 16;

/**
 * BlockTable - Maps sequence positions to physical block indices
 *
 * For a sequence with position p, the block index is:
 *     block_idx = block_table[p / KV_BLOCK_SIZE]
 * The offset within the block is:
 *     offset = p % KV_BLOCK_SIZE
 */
using BlockTable = std::vector<int>;

/**
 * KVCacheBlock - Represents one block in the paged KV cache
 */
struct KVCacheBlock {
  int block_id;   // Unique identifier
  int ref_count;  // Reference count for sharing
  bool is_free;   // Whether block is available
  int num_tokens; // Number of tokens stored (0 to KV_BLOCK_SIZE)

  KVCacheBlock(int id)
      : block_id(id), ref_count(0), is_free(true), num_tokens(0) {}
};

// =============================================================================
// Sequence Types
// =============================================================================

/**
 * SequenceState - Current state of a sequence in the batch
 */
enum class SequenceState {
  WAITING,  // Waiting in queue
  RUNNING,  // Currently being processed
  FINISHED, // Generation complete
  PREEMPTED // Temporarily stopped for higher priority
};

/**
 * SequenceData - Metadata for a single sequence
 */
struct SequenceData {
  int64_t seq_id;             // Unique sequence identifier
  std::vector<int> token_ids; // All tokens (prompt + generated)
  int prompt_len;             // Original prompt length
  int output_len;             // Number of generated tokens
  int max_output_len;         // Maximum tokens to generate
  SequenceState state;        // Current state
  BlockTable block_table;     // Physical block indices

  SequenceData(int64_t id, const std::vector<int> &prompt, int max_len)
      : seq_id(id), token_ids(prompt), prompt_len(prompt.size()), output_len(0),
        max_output_len(max_len), state(SequenceState::WAITING) {}

  int current_len() const { return prompt_len + output_len; }
  bool is_finished() const { return state == SequenceState::FINISHED; }

  int num_blocks_needed() const {
    return (current_len() + KV_BLOCK_SIZE - 1) / KV_BLOCK_SIZE;
  }
};

// =============================================================================
// Attention Types
// =============================================================================

/**
 * AttentionMetadata - Metadata for attention computation
 *
 * This structure is passed to attention kernels to describe the
 * current batch layout and KV cache configuration.
 */
struct AttentionMetadata {
  // Batch information
  int batch_size;  // Number of sequences
  int max_seq_len; // Maximum sequence length in batch
  bool is_prefill; // Prefill phase (true) or decode (false)

  // Sequence lengths (size = batch_size)
  std::vector<int> seq_lens;     // Current length of each sequence
  std::vector<int> context_lens; // Context length (KV cache length)

  // Block tables (size = batch_size x max_blocks)
  std::vector<int> block_tables; // Flattened block tables
  int max_num_blocks;            // Maximum blocks per sequence

  // Prefill-specific
  std::vector<int> query_start_loc; // Start position of each query
  int total_query_len;              // Total query tokens

  // Decode-specific
  std::vector<int> slot_mapping; // Direct slot indices for decode
};

// =============================================================================
// Sampling Types
// =============================================================================

/**
 * SamplingParams - Parameters for token sampling
 */
struct SamplingParams {
  float temperature = 1.0f;        // Temperature for softmax
  float top_p = 1.0f;              // Top-p (nucleus) sampling threshold
  int top_k = -1;                  // Top-k sampling (-1 = disabled)
  float repetition_penalty = 1.0f; // Penalty for repeated tokens
  int max_tokens = 256;            // Maximum tokens to generate
  std::vector<int> stop_token_ids; // Stop generation on these tokens

  bool use_top_k() const { return top_k > 0; }
  bool use_top_p() const { return top_p < 1.0f; }
};

// =============================================================================
// Request Types
// =============================================================================

/**
 * Request - Represents a user request for text generation
 */
struct Request {
  int64_t request_id;             // Unique request identifier
  std::string prompt;             // Input text
  SamplingParams sampling_params; // Sampling configuration
  int64_t arrival_time;           // Timestamp of request arrival

  Request(int64_t id, const std::string &p, const SamplingParams &sp)
      : request_id(id), prompt(p), sampling_params(sp), arrival_time(0) {}
};

/**
 * Response - Generated response for a request
 */
struct Response {
  int64_t request_id;         // Matching request ID
  std::string generated_text; // Generated output
  int num_tokens;             // Number of tokens generated
  float generation_time_ms;   // Total generation time
  bool is_complete;           // Whether generation finished

  Response(int64_t id)
      : request_id(id), num_tokens(0), generation_time_ms(0),
        is_complete(false) {}
};

} // namespace mini_vllm
