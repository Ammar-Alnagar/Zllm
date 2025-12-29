# kv_cache.py - KV Cache Manager for Mini-vLLM

import numpy as np
import torch
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from .config import ModelConfig


@dataclass
class KVCacheConfig:
    """Configuration for KV cache"""

    block_size: int = 16  # Tokens per block
    num_layers: int = 32
    num_kv_heads: int = 8
    head_dim: int = 128
    max_blocks: int = 1000  # Maximum blocks in pool
    dtype: torch.dtype = torch.float16


class BlockAllocator:
    """Block allocator for KV cache"""

    def __init__(self, config: KVCacheConfig):
        self.config = config
        self.free_blocks: List[int] = []
        self.allocated_blocks: Dict[int, int] = {}  # block_id -> ref_count

        # Initialize free list
        for i in range(config.max_blocks):
            self.free_blocks.append(i)

    def allocate_block(self) -> Optional[int]:
        """Allocate a free block"""
        if not self.free_blocks:
            return None

        block_id = self.free_blocks.pop()
        self.allocated_blocks[block_id] = 1
        return block_id

    def deallocate_block(self, block_id: int):
        """Deallocate a block"""
        if block_id in self.allocated_blocks:
            del self.allocated_blocks[block_id]
            self.free_blocks.append(block_id)

    def increase_ref_count(self, block_id: int):
        """Increase reference count for block"""
        if block_id in self.allocated_blocks:
            self.allocated_blocks[block_id] += 1

    def decrease_ref_count(self, block_id: int):
        """Decrease reference count for block"""
        if block_id in self.allocated_blocks:
            self.allocated_blocks[block_id] -= 1
            if self.allocated_blocks[block_id] == 0:
                self.deallocate_block(block_id)

    def get_allocated_blocks(self) -> int:
        """Get number of allocated blocks"""
        return len(self.allocated_blocks)


class KVCacheManager:
    """KV Cache Manager for paged attention"""

    def __init__(self, config: KVCacheConfig):
        self.config = config
        self.allocator = BlockAllocator(config)

        # Block tables: sequence_id -> list of block_ids
        self.block_tables: Dict[int, List[int]] = {}
        self.sequence_lengths: Dict[int, int] = {}

        # GPU memory (if available)
        self.cache_gpu = None
        self._init_gpu_memory()

    def _init_gpu_memory(self):
        """Initialize GPU memory for KV cache"""
        if not torch.cuda.is_available():
            return

        # Calculate memory requirements
        block_size = self.config.block_size
        num_layers = self.config.num_layers
        num_kv_heads = self.config.num_kv_heads
        head_dim = self.config.head_dim
        max_blocks = self.config.max_blocks

        # Shape: [num_blocks, 2, block_size, num_kv_heads, head_dim]
        # 2 for K and V
        cache_shape = (max_blocks, 2, block_size, num_kv_heads, head_dim)

        try:
            self.cache_gpu = torch.zeros(
                cache_shape, dtype=self.config.dtype, device="cuda"
            )
            print(
                f"Allocated {self.cache_gpu.numel() * self.cache_gpu.element_size() / (1024**3):.1f} GB for KV cache"
            )
        except RuntimeError:
            print("Failed to allocate GPU memory for KV cache")
            self.cache_gpu = None

    def allocate_sequence(self, seq_id: int, initial_length: int = 0):
        """Allocate KV cache blocks for a sequence"""
        if seq_id in self.block_tables:
            return  # Already allocated

        self.block_tables[seq_id] = []
        self.sequence_lengths[seq_id] = initial_length

        # Calculate number of blocks needed
        num_blocks_needed = (
            initial_length + self.config.block_size - 1
        ) // self.config.block_size

        for _ in range(num_blocks_needed):
            block_id = self.allocator.allocate_block()
            if block_id is None:
                raise RuntimeError("No free blocks available")
            self.block_tables[seq_id].append(block_id)

    def deallocate_sequence(self, seq_id: int):
        """Deallocate KV cache blocks for a sequence"""
        if seq_id not in self.block_tables:
            return

        # Decrease ref count for all blocks
        for block_id in self.block_tables[seq_id]:
            self.allocator.decrease_ref_count(block_id)

        del self.block_tables[seq_id]
        del self.sequence_lengths[seq_id]

    def get_block_table(self, seq_id: int) -> List[int]:
        """Get block table for sequence"""
        return self.block_tables.get(seq_id, [])

    def get_sequence_length(self, seq_id: int) -> int:
        """Get current length of sequence"""
        return self.sequence_lengths.get(seq_id, 0)

    def update_sequence_length(self, seq_id: int, new_length: int):
        """Update sequence length and allocate more blocks if needed"""
        if seq_id not in self.sequence_lengths:
            raise ValueError(f"Sequence {seq_id} not allocated")

        current_length = self.sequence_lengths[seq_id]
        if new_length <= current_length:
            return  # No need to allocate more blocks

        self.sequence_lengths[seq_id] = new_length

        # Calculate additional blocks needed
        current_blocks = len(self.block_tables[seq_id])
        new_blocks_needed = (
            new_length + self.config.block_size - 1
        ) // self.config.block_size

        for _ in range(current_blocks, new_blocks_needed):
            block_id = self.allocator.allocate_block()
            if block_id is None:
                raise RuntimeError("No free blocks available")
            self.block_tables[seq_id].append(block_id)

    def copy_to_cache(
        self, seq_id: int, layer: int, key: torch.Tensor, value: torch.Tensor
    ):
        """Copy K,V tensors to paged cache"""
        if self.cache_gpu is None:
            return  # CPU fallback

        block_table = self.get_block_table(seq_id)
        seq_len = self.get_sequence_length(seq_id)

        # For simplicity, assume single token append
        # In practice, this would handle batched copying
        if seq_len == 0:
            return

        # Calculate position in cache
        pos = seq_len - 1  # Last token
        block_idx = pos // self.config.block_size
        offset_in_block = pos % self.config.block_size

        if block_idx >= len(block_table):
            return  # Should not happen

        block_id = block_table[block_idx]

        # Copy to GPU cache
        # cache_gpu[block_id, 0, offset_in_block, :, :] = key  # K
        # cache_gpu[block_id, 1, offset_in_block, :, :] = value  # V

        # Note: Actual implementation would use CUDA kernels for efficiency
        self.cache_gpu[block_id, 0, offset_in_block] = key.to(self.cache_gpu.device)
        self.cache_gpu[block_id, 1, offset_in_block] = value.to(self.cache_gpu.device)

    def get_cache_for_inference(
        self, seq_id: int, layer: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get KV cache tensors for inference"""
        if self.cache_gpu is None:
            # CPU fallback - return empty tensors
            return torch.empty(
                0,
                self.config.num_kv_heads,
                self.config.head_dim,
                dtype=self.config.dtype,
            ), torch.empty(
                0,
                self.config.num_kv_heads,
                self.config.head_dim,
                dtype=self.config.dtype,
            )

        block_table = self.get_block_table(seq_id)
        seq_len = self.get_sequence_length(seq_id)

        if not block_table or seq_len == 0:
            return torch.empty(
                0,
                self.config.num_kv_heads,
                self.config.head_dim,
                dtype=self.config.dtype,
            ), torch.empty(
                0,
                self.config.num_kv_heads,
                self.config.head_dim,
                dtype=self.config.dtype,
            )

        # For now, return a simplified view
        # In practice, this would concatenate blocks efficiently
        num_blocks = (seq_len + self.config.block_size - 1) // self.config.block_size

        # Collect blocks
        k_blocks = []
        v_blocks = []

        for i in range(min(num_blocks, len(block_table))):
            block_id = block_table[i]
            k_blocks.append(self.cache_gpu[block_id, 0])  # K
            v_blocks.append(self.cache_gpu[block_id, 1])  # V

        if k_blocks:
            k_cache = torch.cat(k_blocks, dim=0)[:seq_len]
            v_cache = torch.cat(v_blocks, dim=0)[:seq_len]
        else:
            k_cache = torch.empty(
                0,
                self.config.num_kv_heads,
                self.config.head_dim,
                dtype=self.config.dtype,
            )
            v_cache = torch.empty(
                0,
                self.config.num_kv_heads,
                self.config.head_dim,
                dtype=self.config.dtype,
            )

        return k_cache, v_cache

    def get_memory_usage(self) -> Dict[str, float]:
        """Get memory usage statistics"""
        allocated_blocks = self.allocator.get_allocated_blocks()
        total_blocks = self.config.max_blocks

        if self.cache_gpu is not None:
            memory_used = (
                self.cache_gpu.numel() * self.cache_gpu.element_size() / (1024**3)
            )
            memory_total = (
                total_blocks
                * self.cache_gpu[0].numel()
                * self.cache_gpu.element_size()
                / (1024**3)
            )
        else:
            memory_used = 0.0
            memory_total = 0.0

        return {
            "allocated_blocks": allocated_blocks,
            "total_blocks": total_blocks,
            "block_utilization": allocated_blocks / total_blocks
            if total_blocks > 0
            else 0.0,
            "memory_used_gb": memory_used,
            "memory_total_gb": memory_total,
        }
