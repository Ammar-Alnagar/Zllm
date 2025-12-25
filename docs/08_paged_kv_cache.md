# Phase 2: Paged KV Cache Implementation

## Table of Contents

1. [Understanding Paged Memory](#understanding-paged-memory)
2. [KV Cache Design](#kv-cache-design)
3. [Block Allocator](#block-allocator)
4. [CUDA KV Cache Operations](#cuda-kv-cache-operations)
5. [Python Integration](#python-integration)
6. [Testing and Verification](#testing-and-verification)

---

## Understanding Paged Memory

**Paged memory** divides memory into fixed-size blocks, allowing non-contiguous allocation. This is crucial for efficient LLM inference.

```
                    Why Paging?

Without Paging (Contiguous Allocation):
┌────────────────────────────────────────────────────────┐
│  Seq 0 [████████████████████████    ]  max_len reserved│
│  Seq 1 [████████████████████████████████████████    ]  │
│  Seq 2 [████████████    ]                              │
│         ▲              ▲                               │
│         │              └─ Wasted memory!               │
│         └─ Actual tokens                               │
│                                                        │
│  Problem: Must allocate max_len for each sequence      │
│  Memory waste can be 50%+ with variable lengths!       │
└────────────────────────────────────────────────────────┘

With Paging (Block Allocation):
┌────────────────────────────────────────────────────────┐
│  Block Pool:                                           │
│  [S0-B0][S1-B0][S2-B0][S0-B1][S1-B1][S1-B2][S0-B2]... │
│                                                        │
│  Each sequence uses only the blocks it needs           │
│  New blocks allocated on demand                        │
│  Near-zero memory waste!                               │
└────────────────────────────────────────────────────────┘
```

### Benefits of Paging

1. **Memory efficiency**: Allocate only what you need
2. **Prefix sharing**: Multiple sequences can share common blocks
3. **Flexible batching**: Sequences can grow independently
4. **Preemption support**: Easy to free and reallocate

---

## KV Cache Design

### Block Structure

```
                    KV Cache Block Layout

Each block stores KV pairs for BLOCK_SIZE tokens:

Block Layout: [num_blocks, 2, block_size, num_kv_heads, head_dim]
              ───────────┬──┬───────────┬─────────────┬────────
                         │  │           │             │
                         │  │           │             └─ 128 (Qwen3)
                         │  │           └─ 8 (GQA)
                         │  └─ 16 tokens per block
                         └─ K=0, V=1

Single Block Memory:
┌───────────────────────────────────────────────────────────────┐
│                         K Values                              │
│  ┌──────────────────────────────────────────────────────────┐ │
│  │ Token 0  [head0[128], head1[128], ..., head7[128]]       │ │
│  │ Token 1  [head0[128], head1[128], ..., head7[128]]       │ │
│  │ ...                                                       │ │
│  │ Token 15 [head0[128], head1[128], ..., head7[128]]       │ │
│  └──────────────────────────────────────────────────────────┘ │
│                         V Values                              │
│  ┌──────────────────────────────────────────────────────────┐ │
│  │ Token 0  [head0[128], head1[128], ..., head7[128]]       │ │
│  │ Token 1  [head0[128], head1[128], ..., head7[128]]       │ │
│  │ ...                                                       │ │
│  │ Token 15 [head0[128], head1[128], ..., head7[128]]       │ │
│  └──────────────────────────────────────────────────────────┘ │
└───────────────────────────────────────────────────────────────┘

Size per block: 2 × 16 × 8 × 128 × 2 bytes (FP16) = 64 KB
```

### Block Table

```
Block Table: Maps logical blocks → physical blocks

Sequence 0 (45 tokens = 3 blocks):
┌─────────────────────────────────────┐
│  Logical Block │  Physical Block   │
├─────────────────┼───────────────────┤
│       0        │        7          │
│       1        │       23          │
│       2        │       15          │
└─────────────────────────────────────┘

To access token 30 in sequence 0:
  logical_block = 30 // 16 = 1
  block_offset = 30 % 16 = 14
  physical_block = block_table[0][1] = 23
  data = kv_cache[23, :, 14, :, :]
```

---

## Block Allocator

Create file: `mini_vllm/csrc/memory/block_manager.hpp`

```cpp
// =============================================================================
// block_manager.hpp - KV Cache Block Manager
// =============================================================================

#pragma once

#include <vector>
#include <queue>
#include <unordered_map>
#include <unordered_set>
#include <memory>
#include <mutex>
#include <stdexcept>

namespace mini_vllm {

// Block size in tokens
constexpr int KV_BLOCK_SIZE = 16;

/**
 * PhysicalBlock - Represents a block of KV cache memory
 */
struct PhysicalBlock {
    int block_id;           // Unique identifier
    int ref_count;          // Reference count for sharing
    int num_tokens;         // Number of valid tokens (0 to block_size)
    bool is_allocated;      // Whether block is in use

    PhysicalBlock(int id)
        : block_id(id)
        , ref_count(0)
        , num_tokens(0)
        , is_allocated(false) {}
};

/**
 * BlockAllocator - Manages allocation of KV cache blocks
 *
 * Features:
 * - Free list for O(1) allocation
 * - Reference counting for sharing
 * - Support for block recycling
 */
class BlockAllocator {
public:
    /**
     * Constructor
     *
     * @param num_blocks: Total number of blocks in the pool
     * @param block_size: Tokens per block (default 16)
     */
    explicit BlockAllocator(int num_blocks, int block_size = KV_BLOCK_SIZE)
        : num_blocks_(num_blocks)
        , block_size_(block_size)
        , num_free_blocks_(num_blocks) {

        // Initialize all blocks as free
        for (int i = 0; i < num_blocks; i++) {
            blocks_.emplace_back(std::make_unique<PhysicalBlock>(i));
            free_blocks_.push(i);
        }
    }

    /**
     * Get number of free blocks available
     */
    int get_num_free_blocks() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return num_free_blocks_;
    }

    /**
     * Get total number of blocks
     */
    int get_num_total_blocks() const {
        return num_blocks_;
    }

    /**
     * Check if we can allocate n blocks
     */
    bool can_allocate(int num_blocks) const {
        std::lock_guard<std::mutex> lock(mutex_);
        return num_free_blocks_ >= num_blocks;
    }

    /**
     * Allocate a single block
     *
     * @return: Block ID of allocated block
     * @throws: runtime_error if no blocks available
     */
    int allocate() {
        std::lock_guard<std::mutex> lock(mutex_);

        if (free_blocks_.empty()) {
            throw std::runtime_error("No free blocks available");
        }

        int block_id = free_blocks_.front();
        free_blocks_.pop();

        PhysicalBlock* block = blocks_[block_id].get();
        block->is_allocated = true;
        block->ref_count = 1;
        block->num_tokens = 0;

        num_free_blocks_--;
        return block_id;
    }

    /**
     * Allocate multiple blocks
     *
     * @param num_blocks: Number of blocks to allocate
     * @return: Vector of block IDs
     */
    std::vector<int> allocate_n(int num_blocks) {
        std::lock_guard<std::mutex> lock(mutex_);

        if (num_free_blocks_ < num_blocks) {
            throw std::runtime_error(
                "Not enough free blocks: need " + std::to_string(num_blocks) +
                ", have " + std::to_string(num_free_blocks_)
            );
        }

        std::vector<int> allocated;
        allocated.reserve(num_blocks);

        for (int i = 0; i < num_blocks; i++) {
            int block_id = free_blocks_.front();
            free_blocks_.pop();

            PhysicalBlock* block = blocks_[block_id].get();
            block->is_allocated = true;
            block->ref_count = 1;
            block->num_tokens = 0;

            allocated.push_back(block_id);
        }

        num_free_blocks_ -= num_blocks;
        return allocated;
    }

    /**
     * Free a block (decrement reference count)
     *
     * Block is only returned to free list when ref_count reaches 0
     */
    void free(int block_id) {
        std::lock_guard<std::mutex> lock(mutex_);

        if (block_id < 0 || block_id >= num_blocks_) {
            throw std::runtime_error("Invalid block ID: " + std::to_string(block_id));
        }

        PhysicalBlock* block = blocks_[block_id].get();

        if (!block->is_allocated) {
            throw std::runtime_error("Block " + std::to_string(block_id) + " is not allocated");
        }

        block->ref_count--;

        if (block->ref_count == 0) {
            block->is_allocated = false;
            block->num_tokens = 0;
            free_blocks_.push(block_id);
            num_free_blocks_++;
        }
    }

    /**
     * Increase reference count (for sharing)
     */
    void add_ref(int block_id) {
        std::lock_guard<std::mutex> lock(mutex_);

        PhysicalBlock* block = blocks_[block_id].get();
        if (!block->is_allocated) {
            throw std::runtime_error("Cannot add ref to unallocated block");
        }
        block->ref_count++;
    }

    /**
     * Get reference count for a block
     */
    int get_ref_count(int block_id) const {
        std::lock_guard<std::mutex> lock(mutex_);
        return blocks_[block_id]->ref_count;
    }

    /**
     * Update number of tokens in a block
     */
    void set_num_tokens(int block_id, int num_tokens) {
        std::lock_guard<std::mutex> lock(mutex_);
        blocks_[block_id]->num_tokens = num_tokens;
    }

    /**
     * Get number of tokens in a block
     */
    int get_num_tokens(int block_id) const {
        std::lock_guard<std::mutex> lock(mutex_);
        return blocks_[block_id]->num_tokens;
    }

    /**
     * Get memory usage statistics
     */
    void get_stats(int& allocated, int& free, float& utilization) const {
        std::lock_guard<std::mutex> lock(mutex_);
        allocated = num_blocks_ - num_free_blocks_;
        free = num_free_blocks_;
        utilization = 1.0f - static_cast<float>(num_free_blocks_) / num_blocks_;
    }

private:
    int num_blocks_;
    int block_size_;
    int num_free_blocks_;

    std::vector<std::unique_ptr<PhysicalBlock>> blocks_;
    std::queue<int> free_blocks_;

    mutable std::mutex mutex_;
};

/**
 * BlockTable - Maps logical blocks to physical blocks for a sequence
 */
class BlockTable {
public:
    BlockTable() = default;

    void append(int physical_block_id) {
        table_.push_back(physical_block_id);
    }

    int get(int logical_index) const {
        return table_[logical_index];
    }

    void set(int logical_index, int physical_block_id) {
        if (logical_index >= table_.size()) {
            table_.resize(logical_index + 1, -1);
        }
        table_[logical_index] = physical_block_id;
    }

    int size() const {
        return table_.size();
    }

    const std::vector<int>& data() const {
        return table_;
    }

    // For copying to GPU
    int* data_ptr() {
        return table_.data();
    }

private:
    std::vector<int> table_;
};

/**
 * CacheManager - High-level manager for KV cache
 */
class CacheManager {
public:
    CacheManager(
        int num_blocks,
        int block_size,
        int num_layers,
        int num_kv_heads,
        int head_dim
    ) : allocator_(num_blocks, block_size)
      , block_size_(block_size)
      , num_layers_(num_layers)
      , num_kv_heads_(num_kv_heads)
      , head_dim_(head_dim) {}

    /**
     * Allocate blocks for a new sequence
     *
     * @param seq_id: Unique sequence identifier
     * @param num_tokens: Number of tokens to allocate for
     * @return: Block table for the sequence
     */
    BlockTable allocate_sequence(int64_t seq_id, int num_tokens) {
        int num_blocks = (num_tokens + block_size_ - 1) / block_size_;

        BlockTable table;
        std::vector<int> blocks = allocator_.allocate_n(num_blocks);

        for (int block_id : blocks) {
            table.append(block_id);
        }

        sequence_tables_[seq_id] = table;
        return table;
    }

    /**
     * Extend a sequence (allocate more blocks if needed)
     */
    void extend_sequence(int64_t seq_id, int new_num_tokens) {
        BlockTable& table = sequence_tables_[seq_id];
        int current_blocks = table.size();
        int needed_blocks = (new_num_tokens + block_size_ - 1) / block_size_;

        int additional = needed_blocks - current_blocks;
        if (additional > 0) {
            std::vector<int> new_blocks = allocator_.allocate_n(additional);
            for (int block_id : new_blocks) {
                table.append(block_id);
            }
        }
    }

    /**
     * Free all blocks for a sequence
     */
    void free_sequence(int64_t seq_id) {
        auto it = sequence_tables_.find(seq_id);
        if (it != sequence_tables_.end()) {
            for (int block_id : it->second.data()) {
                allocator_.free(block_id);
            }
            sequence_tables_.erase(it);
        }
    }

    /**
     * Get block table for a sequence
     */
    const BlockTable& get_block_table(int64_t seq_id) const {
        return sequence_tables_.at(seq_id);
    }

    BlockAllocator& get_allocator() {
        return allocator_;
    }

private:
    BlockAllocator allocator_;
    int block_size_;
    int num_layers_;
    int num_kv_heads_;
    int head_dim_;

    std::unordered_map<int64_t, BlockTable> sequence_tables_;
};

} // namespace mini_vllm
```

Create file: `mini_vllm/csrc/memory/block_manager.cpp`

```cpp
// =============================================================================
// block_manager.cpp - Block Manager Implementation
// =============================================================================

#include "block_manager.hpp"

namespace mini_vllm {

// Implementation is header-only for this simple case
// Add any non-template implementations here if needed

} // namespace mini_vllm
```

---

## CUDA KV Cache Operations

Create file: `mini_vllm/csrc/memory/kv_cache.cuh`

```cuda
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
```

Create file: `mini_vllm/csrc/memory/kv_cache.cu`

```cuda
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
```

---

## Python Integration

Create file: `mini_vllm/python/mini_vllm/kv_cache.py`

```python
"""
KV Cache Python Interface
=========================

High-level Python wrapper for paged KV cache management.
"""

from dataclasses import dataclass
from typing import List, Optional, Dict, Tuple
import torch


@dataclass
class CacheConfig:
    """Configuration for KV cache"""
    num_blocks: int
    block_size: int = 16
    num_layers: int = 32
    num_kv_heads: int = 8
    head_dim: int = 128
    dtype: torch.dtype = torch.float16
    device: str = "cuda"

    @property
    def kv_cache_shape(self) -> Tuple[int, ...]:
        """Shape of the KV cache tensor"""
        # [num_blocks, 2, block_size, num_kv_heads, head_dim]
        return (
            self.num_blocks,
            2,  # K and V
            self.block_size,
            self.num_kv_heads,
            self.head_dim
        )

    @property
    def bytes_per_block(self) -> int:
        """Memory per block in bytes"""
        elem_size = 2 if self.dtype == torch.float16 else 4
        return 2 * self.block_size * self.num_kv_heads * self.head_dim * elem_size

    @property
    def total_memory_gb(self) -> float:
        """Total cache memory in GB"""
        return (self.num_blocks * self.bytes_per_block) / (1024 ** 3)


class BlockAllocator:
    """
    Manages allocation of KV cache blocks.

    Uses a simple free list for O(1) allocation and deallocation.
    Supports reference counting for prefix sharing.
    """

    def __init__(self, num_blocks: int):
        self.num_blocks = num_blocks

        # Free list (blocks available for allocation)
        self.free_blocks: List[int] = list(range(num_blocks))

        # Reference counts for sharing
        self.ref_counts: Dict[int, int] = {}

    @property
    def num_free(self) -> int:
        return len(self.free_blocks)

    def can_allocate(self, n: int) -> bool:
        return self.num_free >= n

    def allocate(self) -> int:
        """Allocate a single block"""
        if not self.free_blocks:
            raise RuntimeError("No free blocks available")

        block_id = self.free_blocks.pop()
        self.ref_counts[block_id] = 1
        return block_id

    def allocate_n(self, n: int) -> List[int]:
        """Allocate n blocks"""
        if self.num_free < n:
            raise RuntimeError(f"Need {n} blocks, only {self.num_free} available")

        blocks = []
        for _ in range(n):
            blocks.append(self.allocate())
        return blocks

    def free(self, block_id: int):
        """Free a block (decrement ref count)"""
        if block_id not in self.ref_counts:
            raise RuntimeError(f"Block {block_id} not allocated")

        self.ref_counts[block_id] -= 1

        if self.ref_counts[block_id] == 0:
            del self.ref_counts[block_id]
            self.free_blocks.append(block_id)

    def add_ref(self, block_id: int):
        """Increment reference count (for sharing)"""
        if block_id not in self.ref_counts:
            raise RuntimeError(f"Block {block_id} not allocated")
        self.ref_counts[block_id] += 1

    def get_ref_count(self, block_id: int) -> int:
        return self.ref_counts.get(block_id, 0)


class BlockTable:
    """
    Maps logical blocks to physical blocks for a sequence.
    """

    def __init__(self):
        self.table: List[int] = []

    def __len__(self) -> int:
        return len(self.table)

    def __getitem__(self, idx: int) -> int:
        return self.table[idx]

    def append(self, block_id: int):
        self.table.append(block_id)

    def as_tensor(self, device: str = "cuda") -> torch.Tensor:
        return torch.tensor(self.table, dtype=torch.int32, device=device)


class KVCacheManager:
    """
    High-level manager for paged KV cache.

    Handles:
    - Block allocation per sequence
    - Slot mapping for token placement
    - Memory statistics
    """

    def __init__(self, config: CacheConfig):
        self.config = config
        self.allocator = BlockAllocator(config.num_blocks)

        # Allocate the KV cache tensor
        self.kv_cache = torch.zeros(
            config.kv_cache_shape,
            dtype=config.dtype,
            device=config.device
        )

        # Track sequences
        self.sequence_tables: Dict[int, BlockTable] = {}
        self.sequence_lengths: Dict[int, int] = {}

    def allocate_sequence(self, seq_id: int, num_tokens: int) -> BlockTable:
        """
        Allocate blocks for a new sequence.

        Args:
            seq_id: Unique sequence identifier
            num_tokens: Number of tokens to allocate for

        Returns:
            Block table for the sequence
        """
        num_blocks = (num_tokens + self.config.block_size - 1) // self.config.block_size

        blocks = self.allocator.allocate_n(num_blocks)

        table = BlockTable()
        for block_id in blocks:
            table.append(block_id)

        self.sequence_tables[seq_id] = table
        self.sequence_lengths[seq_id] = num_tokens

        return table

    def extend_sequence(self, seq_id: int, additional_tokens: int):
        """
        Extend a sequence by allocating more blocks if needed.
        """
        table = self.sequence_tables[seq_id]
        current_len = self.sequence_lengths[seq_id]
        new_len = current_len + additional_tokens

        current_blocks = len(table)
        needed_blocks = (new_len + self.config.block_size - 1) // self.config.block_size

        additional_blocks = needed_blocks - current_blocks
        if additional_blocks > 0:
            new_blocks = self.allocator.allocate_n(additional_blocks)
            for block_id in new_blocks:
                table.append(block_id)

        self.sequence_lengths[seq_id] = new_len

    def free_sequence(self, seq_id: int):
        """Free all blocks for a sequence"""
        if seq_id in self.sequence_tables:
            table = self.sequence_tables[seq_id]
            for block_id in table.table:
                self.allocator.free(block_id)
            del self.sequence_tables[seq_id]
            del self.sequence_lengths[seq_id]

    def get_slot_mapping(self, seq_id: int, start_pos: int, num_tokens: int) -> torch.Tensor:
        """
        Get slot mapping for token placement.

        Each slot maps to: block_id * block_size + offset
        """
        table = self.sequence_tables[seq_id]

        slots = []
        for i in range(num_tokens):
            pos = start_pos + i
            logical_block = pos // self.config.block_size
            offset = pos % self.config.block_size
            physical_block = table[logical_block]
            slot = physical_block * self.config.block_size + offset
            slots.append(slot)

        return torch.tensor(slots, dtype=torch.int32, device=self.config.device)

    def get_stats(self) -> dict:
        """Get memory usage statistics"""
        return {
            "total_blocks": self.config.num_blocks,
            "free_blocks": self.allocator.num_free,
            "used_blocks": self.config.num_blocks - self.allocator.num_free,
            "utilization": 1.0 - (self.allocator.num_free / self.config.num_blocks),
            "num_sequences": len(self.sequence_tables),
            "total_memory_gb": self.config.total_memory_gb,
        }
```

---

## Testing and Verification

Create file: `mini_vllm/tests/python/test_kv_cache.py`

```python
"""
Test KV Cache Implementation
"""

import pytest
import torch
from mini_vllm.kv_cache import CacheConfig, BlockAllocator, KVCacheManager


class TestBlockAllocator:
    def test_basic_allocation(self):
        allocator = BlockAllocator(num_blocks=100)

        # Allocate single block
        block = allocator.allocate()
        assert block >= 0 and block < 100
        assert allocator.num_free == 99
        assert allocator.get_ref_count(block) == 1

        # Free block
        allocator.free(block)
        assert allocator.num_free == 100
        assert allocator.get_ref_count(block) == 0

    def test_batch_allocation(self):
        allocator = BlockAllocator(num_blocks=100)

        blocks = allocator.allocate_n(10)
        assert len(blocks) == 10
        assert allocator.num_free == 90

        for block in blocks:
            allocator.free(block)
        assert allocator.num_free == 100

    def test_reference_counting(self):
        allocator = BlockAllocator(num_blocks=10)

        block = allocator.allocate()
        assert allocator.get_ref_count(block) == 1

        # Add references (for sharing)
        allocator.add_ref(block)
        allocator.add_ref(block)
        assert allocator.get_ref_count(block) == 3

        # Free requires 3 calls now
        allocator.free(block)
        assert allocator.get_ref_count(block) == 2
        allocator.free(block)
        assert allocator.get_ref_count(block) == 1
        allocator.free(block)
        assert allocator.get_ref_count(block) == 0
        assert allocator.num_free == 10

    def test_allocation_failure(self):
        allocator = BlockAllocator(num_blocks=5)

        allocator.allocate_n(5)

        with pytest.raises(RuntimeError):
            allocator.allocate()


class TestKVCacheManager:
    @pytest.fixture
    def cache_manager(self):
        config = CacheConfig(
            num_blocks=100,
            block_size=16,
            num_layers=32,
            num_kv_heads=8,
            head_dim=128,
            dtype=torch.float16,
            device="cpu"  # Use CPU for tests
        )
        return KVCacheManager(config)

    def test_allocate_sequence(self, cache_manager):
        table = cache_manager.allocate_sequence(seq_id=0, num_tokens=50)

        # 50 tokens @ 16 per block = 4 blocks
        assert len(table) == 4
        assert cache_manager.allocator.num_free == 96

    def test_extend_sequence(self, cache_manager):
        table = cache_manager.allocate_sequence(seq_id=0, num_tokens=16)
        assert len(table) == 1

        # Extend within same block
        cache_manager.extend_sequence(seq_id=0, additional_tokens=10)
        assert len(cache_manager.sequence_tables[0]) == 2  # Now needs 2 blocks

        # Extend to need third block
        cache_manager.extend_sequence(seq_id=0, additional_tokens=20)
        assert len(cache_manager.sequence_tables[0]) == 3

    def test_slot_mapping(self, cache_manager):
        cache_manager.allocate_sequence(seq_id=0, num_tokens=32)

        # Get slots for first 16 tokens (first block)
        slots = cache_manager.get_slot_mapping(seq_id=0, start_pos=0, num_tokens=16)

        table = cache_manager.sequence_tables[0]
        block0 = table[0]

        # Slots should be block0 * 16 + offset
        for i, slot in enumerate(slots.tolist()):
            assert slot == block0 * 16 + i

    def test_free_sequence(self, cache_manager):
        cache_manager.allocate_sequence(seq_id=0, num_tokens=64)
        cache_manager.allocate_sequence(seq_id=1, num_tokens=64)

        assert cache_manager.allocator.num_free == 92

        cache_manager.free_sequence(seq_id=0)
        assert cache_manager.allocator.num_free == 96
        assert 0 not in cache_manager.sequence_tables

    def test_cache_shape(self, cache_manager):
        expected_shape = (100, 2, 16, 8, 128)
        assert cache_manager.kv_cache.shape == expected_shape

    def test_stats(self, cache_manager):
        cache_manager.allocate_sequence(seq_id=0, num_tokens=48)

        stats = cache_manager.get_stats()
        assert stats["total_blocks"] == 100
        assert stats["used_blocks"] == 3
        assert stats["num_sequences"] == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
```

---

## Summary

You've implemented the paged KV cache with:

| Component          | Purpose                               |
| ------------------ | ------------------------------------- |
| **BlockAllocator** | Free list management, O(1) alloc/free |
| **BlockTable**     | Logical → physical block mapping      |
| **CacheManager**   | High-level sequence management        |
| **CUDA kernels**   | Efficient KV copy operations          |

### Key Features

1. **Efficient allocation** - Free list for O(1) operations
2. **Reference counting** - Enables prefix sharing
3. **Vectorized copy** - float4/half2 for bandwidth
4. **Thread-safe** - Mutex protection for concurrent access

---

## What's Next

Next, we'll implement **RadixAttention** for prefix sharing across sequences.

Continue to: [09_radix_attention.md](./09_radix_attention.md)
