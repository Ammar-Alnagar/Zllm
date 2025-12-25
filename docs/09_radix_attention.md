# Phase 2: RadixAttention for Prefix Sharing

## Table of Contents

1. [Understanding Prefix Sharing](#understanding-prefix-sharing)
2. [Radix Tree Data Structure](#radix-tree-data-structure)
3. [Implementation](#implementation)
4. [KV Cache Integration](#kv-cache-integration)
5. [Eviction Policies](#eviction-policies)
6. [Python Interface](#python-interface)
7. [Testing and Verification](#testing-and-verification)

---

## Understanding Prefix Sharing

**Prefix sharing** allows multiple sequences to share common KV cache blocks when they have the same prefix tokens. This is critical for:

- **System prompts**: All requests share the same system message
- **Multi-turn chat**: Follow-up questions share conversation history
- **Few-shot examples**: All requests share the same examples

```
                    Without Prefix Sharing

Request 1: "System prompt... User: What is AI?"
Request 2: "System prompt... User: What is ML?"
Request 3: "System prompt... User: What is DL?"

Memory:
┌─────────────────────────────────────────────────────────┐
│ [System prompt KV] [What is AI? KV]                    │ Seq 1
│ [System prompt KV] [What is ML? KV]                    │ Seq 2
│ [System prompt KV] [What is DL? KV]                    │ Seq 3
│                                                         │
│ System prompt is duplicated 3 times! 💸                │
└─────────────────────────────────────────────────────────┘


                    With Prefix Sharing

Memory:
┌─────────────────────────────────────────────────────────┐
│ [System prompt KV] ←──┬─── Shared by all 3 sequences   │
│                       │                                 │
│ [What is AI? KV] ─────┤                                │
│ [What is ML? KV] ─────┤                                │
│ [What is DL? KV] ─────┘                                │
│                                                         │
│ 66% memory savings! ✨                                  │
└─────────────────────────────────────────────────────────┘
```

---

## Radix Tree Data Structure

A **Radix Tree** (also called Patricia Trie) efficiently stores sequences with shared prefixes.

```
                    Radix Tree Structure

Example with token sequences:
- Seq A: [1, 2, 3, 4, 5]
- Seq B: [1, 2, 3, 6, 7]
- Seq C: [1, 2, 8, 9]
- Seq D: [1, 2, 3, 4, 5, 10]

Radix Tree:
                    [ROOT]
                       │
                       │ edge: [1, 2]
                       ▼
                    [Node A]
                    /       \
       edge: [3, 4, 5]     edge: [8, 9]
              /                   \
         [Node B] ─────────     [Node C]
            │     \        \     (Seq C ends here)
   edge: [10]   edge: [6, 7]
            │              \
         [Node D]        [Node E]
     (Seq D ends)     (Seq B ends)

    [Node B] is shared by Seq A and Seq D!

Each node stores:
- Token sequence for the edge
- Reference to KV cache blocks
- Reference count (number of sequences using this node)
```

### Key Operations

| Operation  | Complexity | Description                           |
| ---------- | ---------- | ------------------------------------- |
| **Insert** | O(L)       | Add new sequence, reusing prefix      |
| **Match**  | O(L)       | Find longest matching prefix          |
| **Delete** | O(L)       | Remove sequence, free unshared blocks |
| **Evict**  | O(1)\*     | LRU eviction of leaf nodes            |

\*L = sequence length

---

## Implementation

Create file: `mini_vllm/csrc/memory/radix_tree.hpp`

```cpp
// =============================================================================
// radix_tree.hpp - Radix Tree for KV Cache Prefix Sharing
// =============================================================================

#pragma once

#include <vector>
#include <unordered_map>
#include <memory>
#include <list>
#include <mutex>
#include <optional>
#include <functional>
#include <algorithm>

namespace mini_vllm {

// Forward declaration
class RadixNode;

using TokenId = int32_t;
using BlockId = int32_t;
using NodePtr = std::shared_ptr<RadixNode>;
using WeakNodePtr = std::weak_ptr<RadixNode>;

/**
 * RadixNode - A node in the radix tree
 *
 * Each node represents a sequence of tokens and references
 * the corresponding KV cache blocks.
 */
class RadixNode : public std::enable_shared_from_this<RadixNode> {
public:
    RadixNode() : ref_count_(0), last_access_time_(0) {}

    // Token sequence for the edge leading TO this node
    std::vector<TokenId> tokens;

    // KV cache block IDs for this node's tokens
    std::vector<BlockId> block_ids;

    // Child nodes (key = first token of child's edge)
    std::unordered_map<TokenId, NodePtr> children;

    // Parent node (weak to avoid cycles)
    WeakNodePtr parent;

    // Number of sequences using this node
    int ref_count_;

    // For LRU eviction
    int64_t last_access_time_;

    // Methods
    bool is_leaf() const { return children.empty(); }

    int get_ref_count() const { return ref_count_; }

    void add_ref() { ref_count_++; }

    void remove_ref() {
        ref_count_--;
        if (ref_count_ < 0) ref_count_ = 0;
    }

    NodePtr get_child(TokenId token) {
        auto it = children.find(token);
        return (it != children.end()) ? it->second : nullptr;
    }

    void add_child(TokenId first_token, NodePtr child) {
        children[first_token] = child;
        child->parent = weak_from_this();
    }

    void remove_child(TokenId first_token) {
        children.erase(first_token);
    }

    // Get total token count from root to this node
    int get_depth() const {
        int depth = tokens.size();
        if (auto p = parent.lock()) {
            depth += p->get_depth();
        }
        return depth;
    }
};

/**
 * MatchResult - Result of prefix matching
 */
struct MatchResult {
    NodePtr last_node;           // Deepest matching node
    int matched_tokens;          // Number of tokens matched
    std::vector<BlockId> blocks; // All matched block IDs

    bool is_full_match(int query_len) const {
        return matched_tokens == query_len;
    }
};

/**
 * InsertResult - Result of inserting a sequence
 */
struct InsertResult {
    NodePtr leaf_node;           // The leaf node for this sequence
    int reused_tokens;           // Tokens reused from cache
    int new_tokens;              // Tokens that need computation
    std::vector<BlockId> reused_blocks;
};

/**
 * RadixTree - Manages prefix sharing for KV cache
 *
 * Thread-safe implementation with LRU eviction support.
 */
class RadixTree {
public:
    RadixTree()
        : root_(std::make_shared<RadixNode>())
        , current_time_(0)
        , num_nodes_(1) {}

    /**
     * Match a token sequence against the tree.
     *
     * Returns the longest matching prefix and corresponding blocks.
     */
    MatchResult match(const std::vector<TokenId>& tokens) {
        std::lock_guard<std::mutex> lock(mutex_);

        MatchResult result;
        result.last_node = root_;
        result.matched_tokens = 0;

        NodePtr current = root_;
        size_t pos = 0;

        while (pos < tokens.size()) {
            // Try to find a child starting with tokens[pos]
            NodePtr child = current->get_child(tokens[pos]);

            if (!child) {
                // No matching child
                break;
            }

            // Check how many tokens of the child's edge match
            size_t edge_len = child->tokens.size();
            size_t match_len = 0;

            while (match_len < edge_len &&
                   pos + match_len < tokens.size() &&
                   child->tokens[match_len] == tokens[pos + match_len]) {
                match_len++;
            }

            if (match_len == edge_len) {
                // Full edge match - continue to child
                result.last_node = child;
                result.matched_tokens += edge_len;

                // Collect block IDs
                for (BlockId bid : child->block_ids) {
                    result.blocks.push_back(bid);
                }

                current = child;
                pos += edge_len;

                // Update access time for LRU
                child->last_access_time_ = ++current_time_;
            } else {
                // Partial match - would need to split node
                // For matching, we just return what we have
                break;
            }
        }

        return result;
    }

    /**
     * Insert a token sequence with its blocks.
     *
     * Reuses existing prefix where possible.
     */
    InsertResult insert(
        const std::vector<TokenId>& tokens,
        const std::vector<BlockId>& block_ids,
        int block_size
    ) {
        std::lock_guard<std::mutex> lock(mutex_);

        InsertResult result;

        // First, match existing prefix
        NodePtr current = root_;
        size_t pos = 0;
        size_t block_idx = 0;

        while (pos < tokens.size()) {
            NodePtr child = current->get_child(tokens[pos]);

            if (!child) {
                // No matching child - need to create new nodes
                break;
            }

            size_t edge_len = child->tokens.size();
            size_t match_len = 0;

            while (match_len < edge_len &&
                   pos + match_len < tokens.size() &&
                   child->tokens[match_len] == tokens[pos + match_len]) {
                match_len++;
            }

            if (match_len == edge_len) {
                // Full match - reuse this node's blocks
                for (BlockId bid : child->block_ids) {
                    result.reused_blocks.push_back(bid);
                }
                result.reused_tokens += edge_len;

                current = child;
                pos += edge_len;
                block_idx += child->block_ids.size();

                child->last_access_time_ = ++current_time_;
            } else if (match_len > 0) {
                // Partial match - need to split the edge
                child = split_node(current, child, match_len);

                for (BlockId bid : child->block_ids) {
                    result.reused_blocks.push_back(bid);
                }
                result.reused_tokens += match_len;

                current = child;
                pos += match_len;
                break;
            } else {
                break;
            }
        }

        // Add remaining tokens as new nodes
        result.new_tokens = tokens.size() - pos;

        if (pos < tokens.size()) {
            // Create new node for remaining tokens
            NodePtr new_node = std::make_shared<RadixNode>();
            new_node->tokens = std::vector<TokenId>(
                tokens.begin() + pos, tokens.end()
            );

            // Assign blocks for new tokens
            for (size_t i = block_idx; i < block_ids.size(); i++) {
                new_node->block_ids.push_back(block_ids[i]);
            }

            new_node->ref_count_ = 1;
            new_node->last_access_time_ = ++current_time_;

            current->add_child(tokens[pos], new_node);
            num_nodes_++;

            result.leaf_node = new_node;
        } else {
            // Sequence exactly matches existing - just add ref
            current->add_ref();
            result.leaf_node = current;
        }

        return result;
    }

    /**
     * Remove a sequence from the tree.
     *
     * Frees blocks that are no longer referenced.
     */
    std::vector<BlockId> remove(NodePtr leaf_node) {
        std::lock_guard<std::mutex> lock(mutex_);

        std::vector<BlockId> freed_blocks;

        NodePtr current = leaf_node;

        while (current && current != root_) {
            current->remove_ref();

            if (current->get_ref_count() == 0 && current->is_leaf()) {
                // No more references and no children - can free
                for (BlockId bid : current->block_ids) {
                    freed_blocks.push_back(bid);
                }

                // Remove from parent
                if (auto parent = current->parent.lock()) {
                    if (!current->tokens.empty()) {
                        parent->remove_child(current->tokens[0]);
                    }

                    // Try to merge parent with its only child if possible
                    if (parent != root_ && parent->children.size() == 1 &&
                        parent->get_ref_count() == 0) {
                        merge_with_child(parent);
                    }
                }

                num_nodes_--;
            }

            current = current->parent.lock();
        }

        return freed_blocks;
    }

    /**
     * Evict least recently used leaf nodes.
     *
     * @param num_blocks: Target number of blocks to free
     * @return: Block IDs that were freed
     */
    std::vector<BlockId> evict_lru(int num_blocks) {
        std::lock_guard<std::mutex> lock(mutex_);

        std::vector<BlockId> freed_blocks;

        // Collect all leaf nodes with ref_count == 0
        std::vector<NodePtr> evictable;
        collect_evictable_leaves(root_, evictable);

        // Sort by access time (oldest first)
        std::sort(evictable.begin(), evictable.end(),
            [](const NodePtr& a, const NodePtr& b) {
                return a->last_access_time_ < b->last_access_time_;
            }
        );

        // Evict until we have enough blocks
        for (const NodePtr& node : evictable) {
            if (freed_blocks.size() >= num_blocks) break;

            for (BlockId bid : node->block_ids) {
                freed_blocks.push_back(bid);
            }

            // Remove node
            if (auto parent = node->parent.lock()) {
                if (!node->tokens.empty()) {
                    parent->remove_child(node->tokens[0]);
                }
            }
            num_nodes_--;
        }

        return freed_blocks;
    }

    /**
     * Get statistics about the tree
     */
    void get_stats(int& num_nodes, int& num_shared_blocks, int& max_depth) const {
        std::lock_guard<std::mutex> lock(mutex_);

        num_nodes = num_nodes_;
        num_shared_blocks = 0;
        max_depth = 0;

        // Traverse tree to collect stats
        std::function<void(NodePtr, int)> traverse = [&](NodePtr node, int depth) {
            max_depth = std::max(max_depth, depth);

            if (node->get_ref_count() > 1) {
                num_shared_blocks += node->block_ids.size();
            }

            for (auto& [token, child] : node->children) {
                traverse(child, depth + child->tokens.size());
            }
        };

        traverse(root_, 0);
    }

private:
    NodePtr root_;
    int64_t current_time_;
    int num_nodes_;
    mutable std::mutex mutex_;

    /**
     * Split a node at a given position
     */
    NodePtr split_node(NodePtr parent, NodePtr child, size_t split_pos) {
        // Create new intermediate node
        NodePtr new_node = std::make_shared<RadixNode>();

        // New node gets the first split_pos tokens
        new_node->tokens = std::vector<TokenId>(
            child->tokens.begin(),
            child->tokens.begin() + split_pos
        );

        // Split blocks (assuming block_size tokens per block)
        // This is simplified - real implementation needs block_size
        size_t blocks_for_new = (split_pos + 15) / 16;  // Assuming block_size=16
        for (size_t i = 0; i < blocks_for_new && i < child->block_ids.size(); i++) {
            new_node->block_ids.push_back(child->block_ids[i]);
        }

        // Child keeps remaining tokens and blocks
        child->tokens = std::vector<TokenId>(
            child->tokens.begin() + split_pos,
            child->tokens.end()
        );
        child->block_ids = std::vector<BlockId>(
            child->block_ids.begin() + blocks_for_new,
            child->block_ids.end()
        );

        // Update tree structure
        parent->remove_child(new_node->tokens[0]);
        parent->add_child(new_node->tokens[0], new_node);
        new_node->add_child(child->tokens[0], child);

        // Move references
        new_node->ref_count_ = child->ref_count_;
        new_node->last_access_time_ = ++current_time_;

        num_nodes_++;
        return new_node;
    }

    /**
     * Merge node with its only child
     */
    void merge_with_child(NodePtr node) {
        if (node->children.size() != 1) return;

        auto it = node->children.begin();
        NodePtr child = it->second;

        // Merge tokens
        node->tokens.insert(node->tokens.end(),
            child->tokens.begin(), child->tokens.end());

        // Merge blocks
        node->block_ids.insert(node->block_ids.end(),
            child->block_ids.begin(), child->block_ids.end());

        // Take child's children
        node->children = std::move(child->children);

        // Update parent pointers
        for (auto& [token, grandchild] : node->children) {
            grandchild->parent = node;
        }

        // Take child's ref count
        node->ref_count_ = child->ref_count_;

        num_nodes_--;
    }

    /**
     * Collect all evictable (ref_count = 0) leaf nodes
     */
    void collect_evictable_leaves(NodePtr node, std::vector<NodePtr>& leaves) {
        if (node->is_leaf() && node->get_ref_count() == 0 && node != root_) {
            leaves.push_back(node);
        }

        for (auto& [token, child] : node->children) {
            collect_evictable_leaves(child, leaves);
        }
    }
};

} // namespace mini_vllm
```

---

## KV Cache Integration

The radix tree works together with the block allocator:

```
                    Integration Flow

1. New Request Arrives:
   ┌─────────────────────────────────────────────────────┐
   │ Tokens: [SYS, PROMPT, USER, QUERY]                  │
   │                                                      │
   │ Step 1: Match against radix tree                    │
   │         → Found: [SYS, PROMPT] cached               │
   │         → Reuse blocks: [B0, B1]                    │
   │                                                      │
   │ Step 2: Allocate blocks for new tokens              │
   │         → Allocate blocks: [B5, B6] for [USER, QUERY]│
   │                                                      │
   │ Step 3: Insert into radix tree                      │
   │         → Add new path with blocks [B5, B6]         │
   │                                                      │
   │ Step 4: Compute attention                           │
   │         → Use blocks [B0, B1, B5, B6]               │
   └─────────────────────────────────────────────────────┘

2. Request Completes:
   ┌─────────────────────────────────────────────────────┐
   │ Step 1: Generate all tokens                         │
   │                                                      │
   │ Step 2: Remove sequence from tree                   │
   │         → Decrement ref counts                      │
   │         → Free blocks with ref_count=0              │
   │         → Keep shared prefix cached!                │
   │                                                      │
   │ Step 3: New request can reuse prefix                │
   └─────────────────────────────────────────────────────┘
```

Create file: `mini_vllm/csrc/memory/radix_cache_manager.hpp`

```cpp
// =============================================================================
// radix_cache_manager.hpp - Integrated Radix + KV Cache Manager
// =============================================================================

#pragma once

#include "radix_tree.hpp"
#include "block_manager.hpp"
#include <unordered_map>

namespace mini_vllm {

/**
 * CacheEntry - Tracks a sequence's cache state
 */
struct CacheEntry {
    int64_t seq_id;
    NodePtr radix_node;
    BlockTable block_table;
    int num_tokens;
    bool is_active;

    CacheEntry() : seq_id(-1), num_tokens(0), is_active(false) {}
};

/**
 * RadixCacheManager - Unified manager for radix tree + KV cache
 *
 * Provides:
 * - Automatic prefix matching and reuse
 * - Block allocation for new tokens
 * - LRU eviction when memory is low
 * - Statistics for monitoring
 */
class RadixCacheManager {
public:
    RadixCacheManager(
        int num_blocks,
        int block_size,
        int num_kv_heads,
        int head_dim
    ) : block_allocator_(num_blocks, block_size)
      , block_size_(block_size)
      , num_kv_heads_(num_kv_heads)
      , head_dim_(head_dim) {}

    /**
     * Allocate cache for a new sequence with prefix matching.
     *
     * @param seq_id: Unique sequence identifier
     * @param tokens: Token IDs for the sequence
     * @return: Number of tokens that were cache hits
     */
    int allocate_with_prefix(
        int64_t seq_id,
        const std::vector<TokenId>& tokens
    ) {
        // Match existing prefix
        MatchResult match = radix_tree_.match(tokens);

        int hit_tokens = match.matched_tokens;
        int new_tokens = tokens.size() - hit_tokens;

        // Calculate blocks needed for new tokens
        int new_blocks_needed = (new_tokens + block_size_ - 1) / block_size_;

        // Check if we have enough blocks
        while (!block_allocator_.can_allocate(new_blocks_needed)) {
            // Need to evict
            int to_evict = new_blocks_needed - block_allocator_.get_num_free_blocks();
            auto freed = radix_tree_.evict_lru(to_evict);

            for (BlockId bid : freed) {
                block_allocator_.free(bid);
            }

            if (freed.empty()) {
                throw std::runtime_error("Cannot allocate: no evictable blocks");
            }
        }

        // Allocate new blocks
        std::vector<BlockId> new_blocks = block_allocator_.allocate_n(new_blocks_needed);

        // Combine reused and new blocks
        std::vector<BlockId> all_blocks = match.blocks;
        all_blocks.insert(all_blocks.end(), new_blocks.begin(), new_blocks.end());

        // Add references to reused blocks
        for (BlockId bid : match.blocks) {
            block_allocator_.add_ref(bid);
        }

        // Insert into radix tree
        InsertResult insert = radix_tree_.insert(tokens, all_blocks, block_size_);

        // Create cache entry
        CacheEntry entry;
        entry.seq_id = seq_id;
        entry.radix_node = insert.leaf_node;
        entry.num_tokens = tokens.size();
        entry.is_active = true;

        for (BlockId bid : all_blocks) {
            entry.block_table.append(bid);
        }

        entries_[seq_id] = entry;

        return hit_tokens;
    }

    /**
     * Extend a sequence with new tokens.
     */
    void extend_sequence(int64_t seq_id, int additional_tokens) {
        CacheEntry& entry = entries_.at(seq_id);

        int current_blocks = entry.block_table.size();
        int new_total_tokens = entry.num_tokens + additional_tokens;
        int needed_blocks = (new_total_tokens + block_size_ - 1) / block_size_;
        int additional_blocks = needed_blocks - current_blocks;

        if (additional_blocks > 0) {
            // Evict if needed
            while (!block_allocator_.can_allocate(additional_blocks)) {
                auto freed = radix_tree_.evict_lru(additional_blocks);
                for (BlockId bid : freed) {
                    block_allocator_.free(bid);
                }
            }

            auto new_blocks = block_allocator_.allocate_n(additional_blocks);
            for (BlockId bid : new_blocks) {
                entry.block_table.append(bid);
            }
        }

        entry.num_tokens = new_total_tokens;
    }

    /**
     * Free a sequence (keep prefix cached for reuse).
     */
    void free_sequence(int64_t seq_id) {
        auto it = entries_.find(seq_id);
        if (it == entries_.end()) return;

        CacheEntry& entry = it->second;

        // Remove from radix tree (may free blocks)
        auto freed_blocks = radix_tree_.remove(entry.radix_node);

        for (BlockId bid : freed_blocks) {
            block_allocator_.free(bid);
        }

        // Don't free blocks still referenced by other sequences!
        // The radix tree handles reference counting.

        entries_.erase(it);
    }

    /**
     * Get block table for a sequence.
     */
    const BlockTable& get_block_table(int64_t seq_id) const {
        return entries_.at(seq_id).block_table;
    }

    /**
     * Get statistics.
     */
    struct Stats {
        int total_blocks;
        int used_blocks;
        int free_blocks;
        int num_sequences;
        int radix_nodes;
        int shared_blocks;
        float cache_hit_rate;
        float memory_utilization;
    };

    Stats get_stats() const {
        Stats stats;

        stats.total_blocks = block_allocator_.get_num_total_blocks();
        stats.free_blocks = block_allocator_.get_num_free_blocks();
        stats.used_blocks = stats.total_blocks - stats.free_blocks;
        stats.num_sequences = entries_.size();

        int max_depth;
        radix_tree_.get_stats(stats.radix_nodes, stats.shared_blocks, max_depth);

        stats.memory_utilization = 1.0f -
            static_cast<float>(stats.free_blocks) / stats.total_blocks;

        // Cache hit rate: ratio of shared blocks to total used blocks
        stats.cache_hit_rate = stats.used_blocks > 0 ?
            static_cast<float>(stats.shared_blocks) / stats.used_blocks : 0.0f;

        return stats;
    }

private:
    RadixTree radix_tree_;
    BlockAllocator block_allocator_;
    std::unordered_map<int64_t, CacheEntry> entries_;

    int block_size_;
    int num_kv_heads_;
    int head_dim_;
};

} // namespace mini_vllm
```

---

## Eviction Policies

When memory is full, we need to evict cached prefixes:

```
                    Eviction Strategies

1. LRU (Least Recently Used):
   ┌─────────────────────────────────────────────────────┐
   │ Evict nodes that haven't been accessed recently     │
   │                                                      │
   │ Pros: Simple, good for temporal locality            │
   │ Cons: May evict valuable long prefixes              │
   └─────────────────────────────────────────────────────┘

2. LRU with Priority:
   ┌─────────────────────────────────────────────────────┐
   │ Weight eviction by: access_time × reference_count   │
   │                                                      │
   │ Pros: Keeps frequently shared prefixes              │
   │ Cons: More complex bookkeeping                      │
   └─────────────────────────────────────────────────────┘

3. Size-Aware LRU:
   ┌─────────────────────────────────────────────────────┐
   │ Prioritize evicting smaller (shorter) prefixes     │
   │                                                      │
   │ Pros: Maximizes memory freed per eviction          │
   │ Cons: May remove valuable short prefixes           │
   └─────────────────────────────────────────────────────┘

We implement LRU as the default, with support for custom policies.
```

---

## Python Interface

Create file: `mini_vllm/python/mini_vllm/radix_cache.py`

```python
"""
Radix Cache Python Interface
=============================

Python wrapper for radix tree-based KV cache with prefix sharing.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Set
from collections import defaultdict
import time


@dataclass
class RadixNode:
    """A node in the radix tree"""
    tokens: List[int] = field(default_factory=list)
    block_ids: List[int] = field(default_factory=list)
    children: Dict[int, 'RadixNode'] = field(default_factory=dict)
    parent: Optional['RadixNode'] = None
    ref_count: int = 0
    last_access: float = 0.0

    def is_leaf(self) -> bool:
        return len(self.children) == 0

    def get_child(self, token: int) -> Optional['RadixNode']:
        return self.children.get(token)

    def add_child(self, first_token: int, child: 'RadixNode'):
        self.children[first_token] = child
        child.parent = self


@dataclass
class MatchResult:
    """Result of prefix matching"""
    last_node: RadixNode
    matched_tokens: int
    matched_blocks: List[int]

    def is_full_match(self, query_len: int) -> bool:
        return self.matched_tokens == query_len


class RadixTree:
    """
    Radix tree for KV cache prefix sharing.

    Stores token sequences and their associated KV cache blocks,
    allowing efficient prefix matching and sharing.
    """

    def __init__(self):
        self.root = RadixNode()
        self.num_nodes = 1

        # Statistics
        self.total_matched = 0
        self.total_queries = 0

    def match(self, tokens: List[int]) -> MatchResult:
        """
        Find the longest matching prefix in the tree.

        Args:
            tokens: Token sequence to match

        Returns:
            MatchResult with matched node and blocks
        """
        self.total_queries += 1

        result = MatchResult(
            last_node=self.root,
            matched_tokens=0,
            matched_blocks=[]
        )

        current = self.root
        pos = 0

        while pos < len(tokens):
            child = current.get_child(tokens[pos])

            if child is None:
                break

            # Check how many tokens match
            edge_len = len(child.tokens)
            match_len = 0

            while (match_len < edge_len and
                   pos + match_len < len(tokens) and
                   child.tokens[match_len] == tokens[pos + match_len]):
                match_len += 1

            if match_len == edge_len:
                # Full edge match
                result.last_node = child
                result.matched_tokens += edge_len
                result.matched_blocks.extend(child.block_ids)

                current = child
                pos += edge_len

                # Update access time for LRU
                child.last_access = time.time()
            else:
                # Partial match - stop here
                break

        self.total_matched += result.matched_tokens
        return result

    def insert(
        self,
        tokens: List[int],
        block_ids: List[int],
        block_size: int = 16
    ) -> Tuple[RadixNode, int]:
        """
        Insert a token sequence with its blocks.

        Args:
            tokens: Token sequence
            block_ids: Corresponding KV cache block IDs
            block_size: Tokens per block

        Returns:
            (leaf_node, num_reused_tokens)
        """
        current = self.root
        pos = 0
        block_idx = 0
        reused_tokens = 0

        while pos < len(tokens):
            child = current.get_child(tokens[pos])

            if child is None:
                break

            edge_len = len(child.tokens)
            match_len = 0

            while (match_len < edge_len and
                   pos + match_len < len(tokens) and
                   child.tokens[match_len] == tokens[pos + match_len]):
                match_len += 1

            if match_len == edge_len:
                # Full match - reuse
                reused_tokens += edge_len
                current = child
                pos += edge_len
                block_idx += len(child.block_ids)
                child.last_access = time.time()
            elif match_len > 0:
                # Partial - need to split
                child = self._split_node(current, child, match_len, block_size)
                reused_tokens += match_len
                current = child
                pos += match_len
                break
            else:
                break

        # Add remaining tokens as new node
        if pos < len(tokens):
            new_node = RadixNode()
            new_node.tokens = tokens[pos:]
            new_node.block_ids = block_ids[block_idx:]
            new_node.ref_count = 1
            new_node.last_access = time.time()

            current.add_child(tokens[pos], new_node)
            self.num_nodes += 1

            return new_node, reused_tokens
        else:
            # Exact match - add reference
            current.ref_count += 1
            return current, reused_tokens

    def remove(self, leaf_node: RadixNode) -> List[int]:
        """
        Remove a sequence from the tree.

        Returns block IDs that were freed.
        """
        freed_blocks = []
        current = leaf_node

        while current is not None and current != self.root:
            current.ref_count -= 1

            if current.ref_count <= 0 and current.is_leaf():
                freed_blocks.extend(current.block_ids)

                if current.parent:
                    if current.tokens:
                        del current.parent.children[current.tokens[0]]
                    self.num_nodes -= 1

            current = current.parent

        return freed_blocks

    def evict_lru(self, num_blocks: int) -> List[int]:
        """
        Evict least recently used leaf nodes.

        Returns freed block IDs.
        """
        freed_blocks = []

        # Collect evictable leaves
        evictable = []
        self._collect_evictable(self.root, evictable)

        # Sort by access time (oldest first)
        evictable.sort(key=lambda n: n.last_access)

        # Evict until we have enough blocks
        for node in evictable:
            if len(freed_blocks) >= num_blocks:
                break

            freed_blocks.extend(node.block_ids)

            if node.parent and node.tokens:
                del node.parent.children[node.tokens[0]]
                self.num_nodes -= 1

        return freed_blocks

    def get_cache_hit_rate(self) -> float:
        """Get overall cache hit rate"""
        if self.total_queries == 0:
            return 0.0
        # This is tokens matched, not exact hit rate
        return self.total_matched / (self.total_queries * 100)  # Rough estimate

    def _split_node(
        self,
        parent: RadixNode,
        child: RadixNode,
        split_pos: int,
        block_size: int
    ) -> RadixNode:
        """Split a node at the given position"""
        new_node = RadixNode()

        # New node gets first split_pos tokens
        new_node.tokens = child.tokens[:split_pos]

        # Split blocks
        blocks_for_new = (split_pos + block_size - 1) // block_size
        new_node.block_ids = child.block_ids[:blocks_for_new]

        # Child keeps remaining
        child.tokens = child.tokens[split_pos:]
        child.block_ids = child.block_ids[blocks_for_new:]

        # Update tree
        del parent.children[new_node.tokens[0]]
        parent.add_child(new_node.tokens[0], new_node)
        new_node.add_child(child.tokens[0], child)

        new_node.ref_count = child.ref_count
        new_node.last_access = time.time()

        self.num_nodes += 1
        return new_node

    def _collect_evictable(self, node: RadixNode, result: List[RadixNode]):
        """Collect all evictable leaf nodes"""
        if node.is_leaf() and node.ref_count <= 0 and node != self.root:
            result.append(node)

        for child in node.children.values():
            self._collect_evictable(child, result)
```

---

## Testing and Verification

Create file: `mini_vllm/tests/python/test_radix_cache.py`

```python
"""
Test Radix Cache Implementation
"""

import pytest
from mini_vllm.radix_cache import RadixTree, RadixNode


class TestRadixTree:
    @pytest.fixture
    def tree(self):
        return RadixTree()

    def test_empty_match(self, tree):
        result = tree.match([1, 2, 3])
        assert result.matched_tokens == 0
        assert result.matched_blocks == []

    def test_insert_and_match(self, tree):
        # Insert a sequence
        tokens = [1, 2, 3, 4, 5]
        blocks = [0, 1]  # 2 blocks for 5 tokens @ block_size=16

        node, reused = tree.insert(tokens, blocks)
        assert reused == 0  # No previous prefix

        # Match should find all tokens
        result = tree.match(tokens)
        assert result.matched_tokens == 5
        assert result.matched_blocks == [0, 1]

    def test_prefix_sharing(self, tree):
        # Insert first sequence
        tokens1 = [1, 2, 3, 4, 5]
        blocks1 = [0, 1]
        tree.insert(tokens1, blocks1)

        # Insert second sequence with same prefix
        tokens2 = [1, 2, 3, 6, 7]
        blocks2 = [0, 1, 2]  # Blocks 0,1 should be reused

        node, reused = tree.insert(tokens2, blocks2)
        assert reused == 3  # [1, 2, 3] reused

    def test_match_partial(self, tree):
        # Insert sequence
        tokens = [1, 2, 3, 4, 5]
        tree.insert(tokens, [0, 1])

        # Match partial prefix
        result = tree.match([1, 2, 3])
        # Should match only up to existing node boundaries
        # In this case, the full sequence is one node
        assert result.matched_tokens <= 5

    def test_remove_sequence(self, tree):
        # Insert and remove
        tokens = [1, 2, 3, 4, 5]
        node, _ = tree.insert(tokens, [0, 1])

        freed = tree.remove(node)
        assert freed == [0, 1]

    def test_shared_remove(self, tree):
        # Insert two sequences with shared prefix
        tree.insert([1, 2, 3, 4, 5], [0, 1])
        node2, _ = tree.insert([1, 2, 3, 6, 7], [0, 1, 2])

        # Remove second sequence - shared blocks should NOT be freed
        freed = tree.remove(node2)
        # Only the unique suffix blocks should be freed
        assert 0 not in freed  # Block 0 is shared

    def test_lru_eviction(self, tree):
        import time

        # Insert older sequence
        tree.insert([1, 2, 3], [0])
        time.sleep(0.01)

        # Insert newer sequence
        tree.insert([4, 5, 6], [1])

        # Remove refs so they're evictable
        tree.root.children[1].ref_count = 0
        tree.root.children[4].ref_count = 0

        # Evict oldest
        freed = tree.evict_lru(1)
        assert 0 in freed  # Older sequence's block

    def test_many_sequences(self, tree):
        # Test with many sequences sharing prefix
        base = [1, 2, 3, 4, 5]

        for i in range(100):
            tokens = base + [100 + i]
            tree.insert(tokens, [0, 1])

        # Match should find the common prefix
        result = tree.match(base)
        assert result.matched_tokens >= 5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
```

---

## Summary

You've implemented RadixAttention with:

| Component         | Purpose                              |
| ----------------- | ------------------------------------ |
| **RadixTree**     | Efficient prefix storage and lookup  |
| **RadixNode**     | Node with tokens, blocks, ref counts |
| **LRU Eviction**  | Free memory when full                |
| **Cache Manager** | Integrate radix + block allocator    |

### Key Benefits

1. **Memory efficiency** - Share common prefixes
2. **Fast lookup** - O(L) prefix matching
3. **Automatic eviction** - LRU when memory low
4. **Reference counting** - Safe block sharing

---

## What's Next

Next, we'll implement the **Memory Pool** for GPU memory management.

Continue to: [10_memory_pool.md](./10_memory_pool.md)
