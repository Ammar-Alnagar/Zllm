# Phase 2: Memory Pool Management

## Table of Contents

1. [GPU Memory Overview](#gpu-memory-overview)
2. [Memory Pool Design](#memory-pool-design)
3. [CUDA Implementation](#cuda-implementation)
4. [Memory Statistics](#memory-statistics)
5. [Python Interface](#python-interface)
6. [Testing and Verification](#testing-and-verification)

---

## GPU Memory Overview

Efficient GPU memory management is critical for high-throughput LLM inference.

```
                    GPU Memory Hierarchy

┌─────────────────────────────────────────────────────────┐
│                    GPU Memory                            │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  ┌──────────────────────────────────────────────────┐   │
│  │            HBM (High Bandwidth Memory)            │   │
│  │                                                    │   │
│  │  Total: 16-80 GB depending on GPU                 │   │
│  │  Bandwidth: 1-3 TB/s                              │   │
│  │                                                    │   │
│  │  Contains:                                         │   │
│  │  ┌───────────────┬───────────────┬────────────┐   │   │
│  │  │ Model Weights │   KV Cache    │  Activations│  │   │
│  │  │  (Fixed)      │  (Dynamic)    │  (Temporary)│  │   │
│  │  │               │               │              │   │   │
│  │  │  ~14 GB (7B)  │  Variable     │  Per-batch  │   │   │
│  │  └───────────────┴───────────────┴────────────┘   │   │
│  │                                                    │   │
│  └──────────────────────────────────────────────────┘   │
│                                                          │
└─────────────────────────────────────────────────────────┘

Memory Budget for RTX 4090 (24 GB):
- Model weights (7B, FP16): ~14 GB
- KV Cache: ~8 GB (configurable)
- Activations + workspace: ~2 GB
```

### Memory Challenges

```
Problem 1: Fragmentation
──────────────────────────
Frequent allocations/frees create unusable gaps:

Before:  [████████░░░░████░░████████░░░░████]
                  ↑       ↑
              Fragmented gaps (unusable)

After pooling: [██████████████████████████████████]
                     Contiguous, efficient


Problem 2: Allocation Overhead
──────────────────────────────
cudaMalloc is SLOW (~100-500 μs per call)

Without pool: malloc → kernel → free → malloc → ...
With pool:    Preallocate once → slice from pool


Problem 3: Peak Memory
──────────────────────
Must handle worst-case concurrent allocations

Solution: Careful memory planning + pooling
```

---

## Memory Pool Design

### Pool Architecture

```
                    Memory Pool Architecture

┌─────────────────────────────────────────────────────────┐
│                    GPU Memory Pool                       │
│                                                          │
│  ┌──────────────────────────────────────────────────┐   │
│  │              KV Cache Pool (Fixed)                │   │
│  │                                                    │   │
│  │  Preallocated at startup                          │   │
│  │  [Block 0][Block 1][Block 2]...[Block N]          │   │
│  │                                                    │   │
│  │  Each block: 64 KB (16 tokens × 8 heads × 128d × 2)│   │
│  │  Total: configurable (e.g., 8 GB = 128K blocks)   │   │
│  └──────────────────────────────────────────────────┘   │
│                                                          │
│  ┌──────────────────────────────────────────────────┐   │
│  │             Activation Pool (Dynamic)             │   │
│  │                                                    │   │
│  │  For intermediate activations during forward pass │   │
│  │  Uses arena allocator with reset per forward      │   │
│  │                                                    │   │
│  │  [──────────── Watermark ──────────────→]         │   │
│  │                                                    │   │
│  └──────────────────────────────────────────────────┘   │
│                                                          │
│  ┌──────────────────────────────────────────────────┐   │
│  │              Workspace Pool (Scratch)             │   │
│  │                                                    │   │
│  │  For cuBLAS, reduction buffers, etc.             │   │
│  │  Small, fixed size                                │   │
│  └──────────────────────────────────────────────────┘   │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

---

## CUDA Implementation

Create file: `mini_vllm/csrc/memory/memory_pool.cuh`

```cuda
// =============================================================================
// memory_pool.cuh - GPU Memory Pool Header
// =============================================================================

#pragma once

#include "common.cuh"
#include <cuda_runtime.h>
#include <vector>
#include <mutex>
#include <unordered_map>

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
        return total_bytes > 0 ?
            static_cast<float>(used_bytes) / total_bytes : 0.0f;
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
    void* allocate(size_t bytes, size_t alignment = 256) {
        // Align offset
        size_t aligned_offset = (offset_ + alignment - 1) & ~(alignment - 1);

        if (aligned_offset + bytes > size_) {
            return nullptr;  // Out of memory
        }

        void* ptr = static_cast<char*>(base_) + aligned_offset;
        offset_ = aligned_offset + bytes;

        if (offset_ > peak_) {
            peak_ = offset_;
        }

        return ptr;
    }

    /**
     * Reset arena (free all allocations at once)
     */
    void reset() {
        offset_ = 0;
    }

    size_t get_used() const { return offset_; }
    size_t get_total() const { return size_; }
    size_t get_peak() const { return peak_; }

private:
    void* base_;
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
    void* get_block_ptr(int block_id) {
        if (block_id < 0 || block_id >= num_blocks_) {
            return nullptr;
        }
        return static_cast<char*>(base_) + block_id * block_size_;
    }

    /**
     * Get base pointer (for passing to kernels)
     */
    void* get_base_ptr() { return base_; }

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
            return blocks;  // Not enough
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
    void free_n(const std::vector<int>& blocks) {
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
        stats.peak_bytes = stats.used_bytes;  // Would need tracking
        stats.num_allocations = 0;  // Would need tracking
        stats.num_frees = 0;

        return stats;
    }

private:
    void* base_;
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
    void init(
        float kv_cache_gb,
        int block_size_tokens,
        int num_kv_heads,
        int head_dim,
        float activation_gb = 1.0f,
        int workspace_mb = 256,
        int dtype_size = 2
    ) {
        // Calculate block size in bytes
        // Block stores K and V: 2 × block_size × num_heads × head_dim × dtype_size
        size_t block_bytes = 2 * block_size_tokens * num_kv_heads * head_dim * dtype_size;

        // Calculate number of blocks
        size_t kv_bytes = static_cast<size_t>(kv_cache_gb * 1024 * 1024 * 1024);
        int num_blocks = kv_bytes / block_bytes;

        // Initialize KV cache pool
        kv_pool_.init(num_blocks, block_bytes);

        // Initialize activation arena
        size_t activation_bytes = static_cast<size_t>(activation_gb * 1024 * 1024 * 1024);
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
    BlockPool& kv_pool() { return kv_pool_; }
    void* kv_cache_ptr() { return kv_pool_.get_base_ptr(); }

    // Activation arena
    void* alloc_activation(size_t bytes) {
        return activation_arena_.allocate(bytes);
    }

    void reset_activations() {
        activation_arena_.reset();
    }

    // Workspace
    void* alloc_workspace(size_t bytes) {
        return workspace_arena_.allocate(bytes);
    }

    void reset_workspace() {
        workspace_arena_.reset();
    }

    // Stats
    void get_memory_stats(PoolStats& kv_stats, PoolStats& act_stats, PoolStats& ws_stats) {
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
        printf("  KV Cache: %.2f GB / %.2f GB (%.1f%%)\n",
               kv.used_bytes / 1e9, kv.total_bytes / 1e9, kv.utilization() * 100);
        printf("  Activations: %.2f MB / %.2f GB (peak: %.2f MB)\n",
               act.used_bytes / 1e6, act.total_bytes / 1e9, act.peak_bytes / 1e6);
        printf("  Workspace: %.2f MB / %.2f MB\n",
               ws.used_bytes / 1e6, ws.total_bytes / 1e6);
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
```

---

## Memory Statistics

Create file: `mini_vllm/csrc/memory/memory_utils.cu`

```cuda
// =============================================================================
// memory_utils.cu - Memory Utility Functions
// =============================================================================

#include "memory_pool.cuh"
#include <cuda_runtime.h>

namespace mini_vllm {

/**
 * Get GPU memory info
 */
void get_gpu_memory_info(size_t& free_bytes, size_t& total_bytes) {
    CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));
}

/**
 * Print detailed GPU memory information
 */
void print_gpu_memory_info() {
    size_t free_bytes, total_bytes;
    get_gpu_memory_info(free_bytes, total_bytes);

    float free_gb = free_bytes / (1024.0f * 1024.0f * 1024.0f);
    float total_gb = total_bytes / (1024.0f * 1024.0f * 1024.0f);
    float used_gb = total_gb - free_gb;

    printf("=== GPU Memory Info ===\n");
    printf("Total: %.2f GB\n", total_gb);
    printf("Used:  %.2f GB (%.1f%%)\n", used_gb, (used_gb / total_gb) * 100);
    printf("Free:  %.2f GB\n", free_gb);
}

/**
 * Calculate maximum KV cache size based on available memory
 *
 * @param model_size_gb: Model weight size in GB
 * @param activation_gb: Reserved for activations
 * @param num_kv_heads: Number of KV heads
 * @param head_dim: Head dimension
 * @param block_size: Tokens per block
 * @param dtype_size: Bytes per element
 * @return: Number of blocks that can be allocated
 */
int calculate_max_kv_blocks(
    float model_size_gb,
    float activation_gb,
    int num_kv_heads,
    int head_dim,
    int block_size,
    int dtype_size
) {
    size_t free_bytes, total_bytes;
    get_gpu_memory_info(free_bytes, total_bytes);

    // Reserve some headroom
    float headroom_gb = 0.5f;

    // Available for KV cache
    float available_gb = (free_bytes / 1e9f) - model_size_gb - activation_gb - headroom_gb;

    if (available_gb < 0) {
        printf("Warning: Not enough GPU memory!\n");
        return 0;
    }

    size_t available_bytes = static_cast<size_t>(available_gb * 1e9);

    // Block size in bytes
    size_t block_bytes = 2 * block_size * num_kv_heads * head_dim * dtype_size;

    int max_blocks = available_bytes / block_bytes;

    printf("Available for KV cache: %.2f GB\n", available_gb);
    printf("Block size: %zu bytes\n", block_bytes);
    printf("Max blocks: %d\n", max_blocks);
    printf("Max tokens: %d\n", max_blocks * block_size);

    return max_blocks;
}

/**
 * Memory-efficient buffer for temporary allocations
 */
class ScratchBuffer {
public:
    ScratchBuffer() : data_(nullptr), size_(0) {}

    ~ScratchBuffer() {
        free();
    }

    void* get(size_t required_size) {
        if (required_size > size_) {
            free();
            CUDA_CHECK(cudaMalloc(&data_, required_size));
            size_ = required_size;
        }
        return data_;
    }

    void free() {
        if (data_) {
            cudaFree(data_);
            data_ = nullptr;
            size_ = 0;
        }
    }

private:
    void* data_;
    size_t size_;
};

// Global scratch buffer (one per stream ideally)
static thread_local ScratchBuffer g_scratch_buffer;

void* get_scratch_buffer(size_t size) {
    return g_scratch_buffer.get(size);
}

} // namespace mini_vllm
```

---

## Python Interface

Create file: `mini_vllm/python/mini_vllm/memory.py`

```python
"""
Memory Management Python Interface
===================================

Python wrapper for GPU memory pool management.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple
import torch


@dataclass
class MemoryConfig:
    """Configuration for GPU memory allocation"""

    # KV Cache settings
    kv_cache_gb: float = 8.0
    block_size: int = 16
    num_kv_heads: int = 8
    head_dim: int = 128
    dtype: torch.dtype = torch.float16

    # Other pools
    activation_gb: float = 1.0
    workspace_mb: int = 256

    @property
    def dtype_size(self) -> int:
        return 2 if self.dtype == torch.float16 else 4

    @property
    def block_bytes(self) -> int:
        """Size of one KV block in bytes"""
        return 2 * self.block_size * self.num_kv_heads * self.head_dim * self.dtype_size

    @property
    def num_blocks(self) -> int:
        """Total number of KV blocks"""
        total_bytes = int(self.kv_cache_gb * 1024 ** 3)
        return total_bytes // self.block_bytes

    @property
    def max_tokens(self) -> int:
        """Maximum tokens that can be cached"""
        return self.num_blocks * self.block_size


@dataclass
class MemoryStats:
    """Memory usage statistics"""

    # KV Cache
    kv_total_gb: float = 0.0
    kv_used_gb: float = 0.0
    kv_blocks_total: int = 0
    kv_blocks_used: int = 0

    # GPU overall
    gpu_total_gb: float = 0.0
    gpu_used_gb: float = 0.0
    gpu_free_gb: float = 0.0

    @property
    def kv_utilization(self) -> float:
        if self.kv_total_gb == 0:
            return 0.0
        return self.kv_used_gb / self.kv_total_gb


class MemoryPool:
    """
    GPU Memory Pool Manager

    Manages pre-allocated GPU memory for KV cache, activations,
    and workspace buffers.
    """

    def __init__(self, config: MemoryConfig, device: str = "cuda"):
        self.config = config
        self.device = device

        # Allocate KV cache
        self.kv_cache = torch.zeros(
            (config.num_blocks, 2, config.block_size,
             config.num_kv_heads, config.head_dim),
            dtype=config.dtype,
            device=device
        )

        # Block tracking
        self.free_blocks: List[int] = list(range(config.num_blocks))
        self.allocated_blocks: Dict[int, int] = {}  # block_id -> ref_count

        # Activation buffer (lazily allocated)
        self._activation_buffer: Optional[torch.Tensor] = None
        self._activation_offset: int = 0

        # Workspace buffer (lazily allocated)
        self._workspace_buffer: Optional[torch.Tensor] = None

    def allocate_block(self) -> int:
        """Allocate a single KV cache block"""
        if not self.free_blocks:
            raise RuntimeError("No free blocks available")

        block_id = self.free_blocks.pop()
        self.allocated_blocks[block_id] = 1
        return block_id

    def allocate_blocks(self, n: int) -> List[int]:
        """Allocate n KV cache blocks"""
        if len(self.free_blocks) < n:
            raise RuntimeError(f"Need {n} blocks, only {len(self.free_blocks)} available")

        blocks = []
        for _ in range(n):
            blocks.append(self.allocate_block())
        return blocks

    def free_block(self, block_id: int):
        """Free a KV cache block"""
        if block_id not in self.allocated_blocks:
            return

        self.allocated_blocks[block_id] -= 1
        if self.allocated_blocks[block_id] <= 0:
            del self.allocated_blocks[block_id]
            self.free_blocks.append(block_id)

    def add_ref(self, block_id: int):
        """Add reference to a block (for sharing)"""
        if block_id in self.allocated_blocks:
            self.allocated_blocks[block_id] += 1

    def get_kv_cache(self) -> torch.Tensor:
        """Get the KV cache tensor"""
        return self.kv_cache

    def get_block_ptr(self, block_id: int) -> torch.Tensor:
        """Get tensor slice for a specific block"""
        return self.kv_cache[block_id]

    def allocate_activation(self, shape: Tuple[int, ...]) -> torch.Tensor:
        """
        Allocate from activation buffer.

        Uses simple bump allocation - call reset_activations() to free.
        """
        size = 1
        for dim in shape:
            size *= dim

        bytes_needed = size * (2 if self.config.dtype == torch.float16 else 4)
        total_bytes = int(self.config.activation_gb * 1024 ** 3)

        if self._activation_buffer is None:
            # Lazy allocation
            num_elements = total_bytes // (2 if self.config.dtype == torch.float16 else 4)
            self._activation_buffer = torch.empty(
                num_elements, dtype=self.config.dtype, device=self.device
            )
            self._activation_offset = 0

        if self._activation_offset + size > len(self._activation_buffer):
            raise RuntimeError("Activation buffer exhausted")

        # Slice from buffer
        tensor = self._activation_buffer[self._activation_offset:self._activation_offset + size]
        tensor = tensor.view(shape)
        self._activation_offset += size

        return tensor

    def reset_activations(self):
        """Reset activation buffer (free all activations)"""
        self._activation_offset = 0

    def get_stats(self) -> MemoryStats:
        """Get memory usage statistics"""
        stats = MemoryStats()

        # KV cache stats
        stats.kv_blocks_total = self.config.num_blocks
        stats.kv_blocks_used = len(self.allocated_blocks)
        stats.kv_total_gb = self.config.kv_cache_gb
        stats.kv_used_gb = (stats.kv_blocks_used / stats.kv_blocks_total) * stats.kv_total_gb

        # GPU stats
        if torch.cuda.is_available():
            stats.gpu_total_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
            allocated = torch.cuda.memory_allocated() / (1024 ** 3)
            reserved = torch.cuda.memory_reserved() / (1024 ** 3)
            stats.gpu_used_gb = reserved
            stats.gpu_free_gb = stats.gpu_total_gb - stats.gpu_used_gb

        return stats

    def print_stats(self):
        """Print memory statistics"""
        stats = self.get_stats()

        print("=== Memory Pool Stats ===")
        print(f"KV Cache: {stats.kv_used_gb:.2f} / {stats.kv_total_gb:.2f} GB "
              f"({stats.kv_utilization * 100:.1f}%)")
        print(f"KV Blocks: {stats.kv_blocks_used} / {stats.kv_blocks_total}")
        print(f"GPU Total: {stats.gpu_total_gb:.2f} GB")
        print(f"GPU Used: {stats.gpu_used_gb:.2f} GB")
        print(f"GPU Free: {stats.gpu_free_gb:.2f} GB")


def estimate_memory_requirements(
    model_size_b: float,
    max_context: int,
    max_batch: int,
    num_kv_heads: int = 8,
    head_dim: int = 128,
    num_layers: int = 32,
    dtype_bytes: int = 2
) -> Dict[str, float]:
    """
    Estimate GPU memory requirements.

    Args:
        model_size_b: Model size in billions of parameters
        max_context: Maximum context length
        max_batch: Maximum batch size
        num_kv_heads: Number of KV heads per layer
        head_dim: Dimension per head
        num_layers: Number of transformer layers
        dtype_bytes: Bytes per element (2 for FP16)

    Returns:
        Dictionary with memory estimates in GB
    """
    # Model weights
    model_gb = model_size_b * 2 * dtype_bytes  # 2 bytes per param for FP16

    # KV cache per token
    kv_per_token = 2 * num_kv_heads * head_dim * num_layers * dtype_bytes

    # Total KV for max batch and context
    kv_gb = (max_batch * max_context * kv_per_token) / (1024 ** 3)

    # Activations (rough estimate: ~2x hidden size per token)
    hidden_size = num_kv_heads * head_dim  # Approximate (should use num_heads)
    activation_per_token = hidden_size * 4 * dtype_bytes  # Factor of 4 for intermediate
    activation_gb = (max_batch * max_context * activation_per_token) / (1024 ** 3)

    # Workspace
    workspace_gb = 0.5

    return {
        "model_weights_gb": model_gb,
        "kv_cache_gb": kv_gb,
        "activations_gb": activation_gb,
        "workspace_gb": workspace_gb,
        "total_gb": model_gb + kv_gb + activation_gb + workspace_gb,
        "kv_per_token_bytes": kv_per_token,
    }
```

---

## Testing and Verification

Create file: `mini_vllm/tests/python/test_memory.py`

```python
"""
Test Memory Pool Implementation
"""

import pytest
import torch
from mini_vllm.memory import MemoryConfig, MemoryPool, estimate_memory_requirements


class TestMemoryPool:
    @pytest.fixture
    def memory_config(self):
        return MemoryConfig(
            kv_cache_gb=1.0,  # Small for testing
            block_size=16,
            num_kv_heads=8,
            head_dim=128,
            dtype=torch.float16,
        )

    @pytest.fixture
    def memory_pool(self, memory_config):
        # Use CPU for tests
        return MemoryPool(memory_config, device="cpu")

    def test_block_allocation(self, memory_pool):
        block = memory_pool.allocate_block()
        assert block >= 0 and block < memory_pool.config.num_blocks
        assert len(memory_pool.free_blocks) == memory_pool.config.num_blocks - 1

    def test_batch_allocation(self, memory_pool):
        blocks = memory_pool.allocate_blocks(10)
        assert len(blocks) == 10
        assert len(memory_pool.free_blocks) == memory_pool.config.num_blocks - 10

    def test_free_block(self, memory_pool):
        block = memory_pool.allocate_block()
        memory_pool.free_block(block)
        assert len(memory_pool.free_blocks) == memory_pool.config.num_blocks

    def test_reference_counting(self, memory_pool):
        block = memory_pool.allocate_block()
        memory_pool.add_ref(block)
        memory_pool.add_ref(block)

        assert memory_pool.allocated_blocks[block] == 3

        memory_pool.free_block(block)
        memory_pool.free_block(block)
        assert memory_pool.allocated_blocks[block] == 1

        memory_pool.free_block(block)
        assert block not in memory_pool.allocated_blocks

    def test_kv_cache_shape(self, memory_pool):
        config = memory_pool.config
        expected_shape = (
            config.num_blocks, 2, config.block_size,
            config.num_kv_heads, config.head_dim
        )
        assert memory_pool.kv_cache.shape == expected_shape

    def test_stats(self, memory_pool):
        memory_pool.allocate_blocks(5)
        stats = memory_pool.get_stats()

        assert stats.kv_blocks_used == 5
        assert stats.kv_utilization > 0


class TestMemoryEstimation:
    def test_7b_model(self):
        estimates = estimate_memory_requirements(
            model_size_b=7,
            max_context=4096,
            max_batch=32,
            num_kv_heads=8,
            head_dim=128,
            num_layers=32,
        )

        # 7B model should need ~14GB for weights
        assert 10 < estimates["model_weights_gb"] < 20

        # KV cache should be significant
        assert estimates["kv_cache_gb"] > 1

    def test_smaller_context(self):
        large_ctx = estimate_memory_requirements(
            model_size_b=7, max_context=8192, max_batch=1,
            num_kv_heads=8, head_dim=128, num_layers=32
        )

        small_ctx = estimate_memory_requirements(
            model_size_b=7, max_context=1024, max_batch=1,
            num_kv_heads=8, head_dim=128, num_layers=32
        )

        # Larger context should need more KV cache
        assert large_ctx["kv_cache_gb"] > small_ctx["kv_cache_gb"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
```

---

## Summary

You've implemented GPU memory management with:

| Component             | Purpose                                  |
| --------------------- | ---------------------------------------- |
| **BlockPool**         | Fixed-size block allocation for KV cache |
| **ArenaAllocator**    | Bump allocator for activations           |
| **MemoryManager**     | Unified interface for all pools          |
| **Memory estimation** | Calculate requirements before running    |

### Key Features

1. **Zero fragmentation** - Pre-allocated pools
2. **Fast allocation** - O(1) block alloc/free
3. **Efficient reset** - Arena clears instantly
4. **Statistics** - Monitor memory usage

---

## What's Next

Now we move to **Phase 3: Python Integration** - building the scheduler and inference engine.

Continue to: [11_scheduler.md](./11_scheduler.md)
