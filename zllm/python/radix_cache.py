# radix_cache.py - Radix Tree Cache for Prefix Sharing

from typing import Dict, List, Optional, Set, Tuple
import hashlib


class RadixNode:
    """Node in the radix tree for prefix sharing"""

    def __init__(self, token: Optional[int] = None):
        self.token = token
        self.children: Dict[int, "RadixNode"] = {}
        self.ref_count = 0
        self.kv_blocks: List[int] = []  # Block IDs that contain this sequence
        self.sequence_ids: Set[int] = set()  # Sequences that share this prefix

    def is_leaf(self) -> bool:
        """Check if this is a leaf node"""
        return len(self.children) == 0


class RadixCache:
    """Radix tree for efficient prefix sharing in KV cache"""

    def __init__(self, block_size: int = 16):
        """Initialize radix cache

        Args:
            block_size: Tokens per KV cache block
        """
        self.block_size = block_size
        self.root = RadixNode()
        self.sequence_prefixes: Dict[int, List[int]] = {}  # seq_id -> token sequence
        self.node_cache: Dict[str, RadixNode] = {}  # hash -> node for fast lookup

    def insert_sequence(self, seq_id: int, tokens: List[int]) -> List[int]:
        """Insert a sequence into the radix tree

        Args:
            seq_id: Sequence identifier
            tokens: Token sequence

        Returns:
            List of block IDs that can be shared
        """
        self.sequence_prefixes[seq_id] = tokens.copy()

        # Traverse/create nodes for this sequence
        current_node = self.root
        shared_blocks = []

        for i, token in enumerate(tokens):
            if token not in current_node.children:
                current_node.children[token] = RadixNode(token)

            current_node = current_node.children[token]
            current_node.ref_count += 1
            current_node.sequence_ids.add(seq_id)

            # Check if we can share a complete block
            if (i + 1) % self.block_size == 0:
                block_start = i - self.block_size + 1
                block_tokens = tokens[block_start : i + 1]

                # Check if this block is already cached
                block_hash = self._hash_tokens(block_tokens)
                if block_hash in self.node_cache:
                    cached_node = self.node_cache[block_hash]
                    if cached_node.kv_blocks:
                        shared_blocks.extend(cached_node.kv_blocks)
                else:
                    self.node_cache[block_hash] = current_node
                    # In a real implementation, we'd allocate blocks here
                    # For now, just track that this block exists
                    current_node.kv_blocks.append(len(shared_blocks))

        return shared_blocks

    def remove_sequence(self, seq_id: int):
        """Remove a sequence from the radix tree"""
        if seq_id not in self.sequence_prefixes:
            return

        tokens = self.sequence_prefixes[seq_id]

        # Traverse and decrement reference counts
        current_node = self.root
        path = [current_node]

        for token in tokens:
            if token in current_node.children:
                current_node = current_node.children[token]
                path.append(current_node)
            else:
                break

        # Decrement ref counts in reverse order
        for node in reversed(path):
            node.ref_count -= 1
            if seq_id in node.sequence_ids:
                node.sequence_ids.remove(seq_id)

        del self.sequence_prefixes[seq_id]

    def find_shared_prefix(
        self, seq_id: int, new_tokens: List[int]
    ) -> Tuple[List[int], int]:
        """Find the longest shared prefix for new tokens

        Args:
            seq_id: Sequence identifier
            new_tokens: New tokens to append

        Returns:
            Tuple of (shared_block_ids, shared_length)
        """
        if seq_id not in self.sequence_prefixes:
            return [], 0

        existing_tokens = self.sequence_prefixes[seq_id]
        full_sequence = existing_tokens + new_tokens

        # Find longest matching prefix in the tree
        current_node = self.root
        shared_length = 0
        shared_blocks = []

        for i, token in enumerate(full_sequence):
            if token in current_node.children:
                current_node = current_node.children[token]
                shared_length = i + 1

                # Check for complete blocks
                if shared_length % self.block_size == 0:
                    block_tokens = full_sequence[
                        shared_length - self.block_size : shared_length
                    ]
                    block_hash = self._hash_tokens(block_tokens)
                    if block_hash in self.node_cache:
                        cached_node = self.node_cache[block_hash]
                        shared_blocks.extend(cached_node.kv_blocks)
            else:
                break

        return shared_blocks, shared_length

    def get_memory_usage(self) -> Dict[str, int]:
        """Get memory usage statistics"""
        total_nodes = self._count_nodes(self.root)
        total_sequences = len(self.sequence_prefixes)

        return {
            "total_nodes": total_nodes,
            "total_sequences": total_sequences,
            "average_sequence_length": sum(
                len(seq) for seq in self.sequence_prefixes.values()
            )
            / max(1, total_sequences),
            "cached_blocks": len(self.node_cache),
        }

    def _count_nodes(self, node: RadixNode) -> int:
        """Count total nodes in subtree"""
        count = 1  # Count this node
        for child in node.children.values():
            count += self._count_nodes(child)
        return count

    def _hash_tokens(self, tokens: List[int]) -> str:
        """Create hash for token sequence"""
        token_bytes = ",".join(map(str, tokens)).encode("utf-8")
        return hashlib.md5(token_bytes).hexdigest()

    def clear(self):
        """Clear the radix tree"""
        self.root = RadixNode()
        self.sequence_prefixes.clear()
        self.node_cache.clear()

    def __len__(self) -> int:
        """Number of sequences in cache"""
        return len(self.sequence_prefixes)
