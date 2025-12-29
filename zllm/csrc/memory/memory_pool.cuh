// =============================================================================
// memory_pool.cuh - GPU Memory Pool Header
// =============================================================================

#pragma once

#include "common.cuh"
#include <cuda_runtime.h>
#include <mutex>
#include <unordered_map>
#include <vector>

namespace mini_vllm {

/**
 * PoolStats - Memory pool statistics
 */
struct PoolStats {
  size_t total_bytes;
  size_t used_bytes;
  size_t peak_bytes;
  size_t num_allocations;
  size_t num_frees;

  float utilization() const {
    return total_bytes > 0 ? static_cast<float>(used_bytes) / total_bytes
                           : 0.0f;
  }
};

/**
 * ArenaAllocator - Simple bump allocator for temporary memory
 *
 * Allocates by moving a pointer forward. Reset clears everything.
 * Very fast but no individual frees.
 */
class ArenaAllocator {
public:
  ArenaAllocator() : base_(nullptr), size_(0), offset_(0), peak_(0) {}

  ~ArenaAllocator() {
    if (base_) {
      cudaFree(base_);
    }
  }

  /**
   * Initialize arena with given size
   */
  void init(size_t size) {
    if (base_) {
      cudaFree(base_);
    }

    CUDA_CHECK(cudaMalloc(&base_, size));
    size_ = size;
    offset_ = 0;
    peak_ = 0;
  }

  /**
   * Allocate from arena
   *
   * @param bytes: Number of bytes to allocate
   * @param alignment: Alignment requirement (default 256 for coalescing)
   * @return: Device pointer, or nullptr if out of space
   */
  void *allocate(size_t bytes, size_t alignment = 256) {
    // Align offset
    size_t aligned_offset = (offset_ + alignment - 1) & ~(alignment - 1);

    if (aligned_offset + bytes > size_) {
      return nullptr; // Out of memory
    }

    void *ptr = static_cast<char *>(base_) + aligned_offset;
    offset_ = aligned_offset + bytes;

    if (offset_ > peak_) {
      peak_ = offset_;
    }

    return ptr;
  }

  /**
   * Reset arena (free all allocations at once)
   */
  void reset() { offset_ = 0; }

  size_t get_used() const { return offset_; }
  size_t get_total() const { return size_; }
  size_t get_peak() const { return peak_; }

private:
  void *base_;
  size_t size_;
  size_t offset_;
  size_t peak_;
};

/**
 * BlockPool - Fixed-size block allocator
 *
 * For KV cache blocks. All blocks are same size.
 */
class BlockPool {
public:
  BlockPool() : base_(nullptr), num_blocks_(0), block_size_(0) {}

  ~BlockPool() {
    if (base_) {
      cudaFree(base_);
    }
  }

  /**
   * Initialize pool
   *
   * @param num_blocks: Number of blocks
   * @param block_size: Size of each block in bytes
   */
  void init(int num_blocks, size_t block_size) {
    if (base_) {
      cudaFree(base_);
    }

    size_t total = static_cast<size_t>(num_blocks) * block_size;
    CUDA_CHECK(cudaMalloc(&base_, total));

    num_blocks_ = num_blocks;
    block_size_ = block_size;

    // All blocks start as free
    free_list_.clear();
    for (int i = 0; i < num_blocks; i++) {
      free_list_.push_back(i);
    }

    allocated_.clear();
  }

  /**
   * Get pointer to a specific block
   */
  void *get_block_ptr(int block_id) {
    if (block_id < 0 || block_id >= num_blocks_) {
      return nullptr;
    }
    return static_cast<char *>(base_) + block_id * block_size_;
  }

  /**
   * Get base pointer (for passing to kernels)
   */
  void *get_base_ptr() { return base_; }

  /**
   * Allocate a block
   *
   * @return: Block ID, or -1 if none available
   */
  int allocate() {
    std::lock_guard<std::mutex> lock(mutex_);

    if (free_list_.empty()) {
      return -1;
    }

    int block_id = free_list_.back();
    free_list_.pop_back();
    allocated_.insert(block_id);

    return block_id;
  }

  /**
   * Allocate multiple blocks
   */
  std::vector<int> allocate_n(int n) {
    std::lock_guard<std::mutex> lock(mutex_);

    std::vector<int> blocks;

    if (free_list_.size() < n) {
      return blocks; // Not enough
    }

    for (int i = 0; i < n; i++) {
      int block_id = free_list_.back();
      free_list_.pop_back();
      allocated_.insert(block_id);
      blocks.push_back(block_id);
    }

    return blocks;
  }

  /**
   * Free a block
   */
  void free(int block_id) {
    std::lock_guard<std::mutex> lock(mutex_);

    auto it = allocated_.find(block_id);
    if (it != allocated_.end()) {
      allocated_.erase(it);
      free_list_.push_back(block_id);
    }
  }

  /**
   * Free multiple blocks
   */
  void free_n(const std::vector<int> &blocks) {
    std::lock_guard<std::mutex> lock(mutex_);

    for (int block_id : blocks) {
      auto it = allocated_.find(block_id);
      if (it != allocated_.end()) {
        allocated_.erase(it);
        free_list_.push_back(block_id);
      }
    }
  }

  int num_free() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return free_list_.size();
  }

  int num_allocated() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return allocated_.size();
  }

  int num_total() const { return num_blocks_; }
  size_t block_size() const { return block_size_; }

  PoolStats get_stats() const {
    std::lock_guard<std::mutex> lock(mutex_);

    PoolStats stats;
    stats.total_bytes = static_cast<size_t>(num_blocks_) * block_size_;
    stats.used_bytes = allocated_.size() * block_size_;
    stats.peak_bytes = stats.used_bytes; // Would need tracking
    stats.num_allocations = 0;           // Would need tracking
    stats.num_frees = 0;

    return stats;
  }

private:
  void *base_;
  int num_blocks_;
  size_t block_size_;

  std::vector<int> free_list_;
  std::unordered_set<int> allocated_;
  mutable std::mutex mutex_;
};

/**
 * MemoryManager - Unified GPU memory manager
 *
 * Manages all GPU memory pools for the inference engine.
 */
class MemoryManager {
public:
  MemoryManager() = default;

  /**
   * Initialize memory manager
   *
   * @param kv_cache_gb: GB for KV cache
   * @param block_size_tokens: Tokens per KV block
   * @param num_kv_heads: Number of KV heads
   * @param head_dim: Head dimension
   * @param activation_gb: GB for activation arena
   * @param workspace_mb: MB for workspace
   * @param dtype_size: Bytes per element (2 for FP16, 4 for FP32)
   */
  void init(float kv_cache_gb, int block_size_tokens, int num_kv_heads,
            int head_dim, float activation_gb = 1.0f, int workspace_mb = 256,
            int dtype_size = 2) {
    // Calculate block size in bytes
    // Block stores K and V: 2 × block_size × num_heads × head_dim × dtype_size
    size_t block_bytes =
        2 * block_size_tokens * num_kv_heads * head_dim * dtype_size;

    // Calculate number of blocks
    size_t kv_bytes = static_cast<size_t>(kv_cache_gb * 1024 * 1024 * 1024);
    int num_blocks = kv_bytes / block_bytes;

    // Initialize KV cache pool
    kv_pool_.init(num_blocks, block_bytes);

    // Initialize activation arena
    size_t activation_bytes =
        static_cast<size_t>(activation_gb * 1024 * 1024 * 1024);
    activation_arena_.init(activation_bytes);

    // Initialize workspace
    size_t workspace_bytes = static_cast<size_t>(workspace_mb) * 1024 * 1024;
    workspace_arena_.init(workspace_bytes);

    block_size_tokens_ = block_size_tokens;
    num_kv_heads_ = num_kv_heads;
    head_dim_ = head_dim;
    dtype_size_ = dtype_size;
  }

  // KV Cache access
  BlockPool &kv_pool() { return kv_pool_; }
  void *kv_cache_ptr() { return kv_pool_.get_base_ptr(); }

  // Activation arena
  void *alloc_activation(size_t bytes) {
    return activation_arena_.allocate(bytes);
  }

  void reset_activations() { activation_arena_.reset(); }

  // Workspace
  void *alloc_workspace(size_t bytes) {
    return workspace_arena_.allocate(bytes);
  }

  void reset_workspace() { workspace_arena_.reset(); }

  // Stats
  void get_memory_stats(PoolStats &kv_stats, PoolStats &act_stats,
                        PoolStats &ws_stats) {
    kv_stats = kv_pool_.get_stats();

    act_stats.total_bytes = activation_arena_.get_total();
    act_stats.used_bytes = activation_arena_.get_used();
    act_stats.peak_bytes = activation_arena_.get_peak();

    ws_stats.total_bytes = workspace_arena_.get_total();
    ws_stats.used_bytes = workspace_arena_.get_used();
    ws_stats.peak_bytes = workspace_arena_.get_peak();
  }

  void print_stats() {
    PoolStats kv, act, ws;
    get_memory_stats(kv, act, ws);

    printf("GPU Memory Usage:\n");
    printf("  KV Cache: %.2f GB / %.2f GB (%.1f%%)\n", kv.used_bytes / 1e9,
           kv.total_bytes / 1e9, kv.utilization() * 100);
    printf("  Activations: %.2f MB / %.2f GB (peak: %.2f MB)\n",
           act.used_bytes / 1e6, act.total_bytes / 1e9, act.peak_bytes / 1e6);
    printf("  Workspace: %.2f MB / %.2f MB\n", ws.used_bytes / 1e6,
           ws.total_bytes / 1e6);
  }

private:
  BlockPool kv_pool_;
  ArenaAllocator activation_arena_;
  ArenaAllocator workspace_arena_;

  int block_size_tokens_;
  int num_kv_heads_;
  int head_dim_;
  int dtype_size_;
};

} // namespace mini_vllm
