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