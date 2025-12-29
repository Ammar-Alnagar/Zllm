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