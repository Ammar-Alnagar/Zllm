# memory.py - Memory Pool Manager for Mini-vLLM

import torch
from typing import Optional, Dict, Any
from contextlib import contextmanager
from dataclasses import dataclass


@dataclass
class MemoryStats:
    """Memory usage statistics"""

    allocated_bytes: int = 0
    peak_bytes: int = 0
    num_allocations: int = 0
    cache_hits: int = 0
    cache_misses: int = 0


class MemoryPool:
    """GPU memory pool for efficient allocation"""

    def __init__(self, device: str = "cuda"):
        self.device = device
        self.pool: Dict[int, list] = {}  # size -> list of tensors
        self.stats = MemoryStats()
        self.enabled = torch.cuda.is_available() and device == "cuda"

    def allocate(
        self, size_bytes: int, dtype: torch.dtype = torch.float16
    ) -> torch.Tensor:
        """Allocate tensor from pool or create new one"""
        if not self.enabled:
            # Fallback to direct allocation
            element_size = torch.tensor([], dtype=dtype).element_size()
            num_elements = size_bytes // element_size
            return torch.empty(num_elements, dtype=dtype, device=self.device)

        # Calculate number of elements
        element_size = torch.tensor([], dtype=dtype).element_size()
        num_elements = size_bytes // element_size

        # Try to reuse from pool
        if size_bytes in self.pool and self.pool[size_bytes]:
            tensor = self.pool[size_bytes].pop()
            self.stats.cache_hits += 1
            return tensor

        # Create new tensor
        tensor = torch.empty(num_elements, dtype=dtype, device=self.device)
        self.stats.allocated_bytes += size_bytes
        self.stats.peak_bytes = max(self.stats.peak_bytes, self.stats.allocated_bytes)
        self.stats.num_allocations += 1
        self.stats.cache_misses += 1

        return tensor

    def free(self, tensor: torch.Tensor):
        """Return tensor to pool for reuse"""
        if not self.enabled or not tensor.is_cuda:
            return

        size_bytes = tensor.numel() * tensor.element_size()

        # Clear tensor contents (optional, for safety)
        tensor.zero_()

        # Add to pool
        if size_bytes not in self.pool:
            self.pool[size_bytes] = []
        self.pool[size_bytes].append(tensor)

        self.stats.allocated_bytes -= size_bytes

    def clear_pool(self):
        """Clear the memory pool"""
        self.pool.clear()
        self.stats = MemoryStats()

    def get_stats(self) -> MemoryStats:
        """Get memory statistics"""
        return MemoryStats(
            allocated_bytes=self.stats.allocated_bytes,
            peak_bytes=self.stats.peak_bytes,
            num_allocations=self.stats.num_allocations,
            cache_hits=self.stats.cache_hits,
            cache_misses=self.stats.cache_misses,
        )

    def get_cache_hit_rate(self) -> float:
        """Get cache hit rate"""
        total = self.stats.cache_hits + self.stats.cache_misses
        return self.stats.cache_hits / total if total > 0 else 0.0


@contextmanager
def memory_pool_context(device: str = "cuda"):
    """Context manager for memory pool"""
    pool = MemoryPool(device)
    try:
        yield pool
    finally:
        pool.clear_pool()


class GPUMemoryManager:
    """Advanced GPU memory manager with defragmentation"""

    def __init__(self, device: str = "cuda"):
        self.device = device
        self.pool = MemoryPool(device)
        self.allocated_tensors: Dict[str, torch.Tensor] = {}
        self.enabled = torch.cuda.is_available() and device == "cuda"

    def allocate_persistent(
        self, name: str, size_bytes: int, dtype: torch.dtype = torch.float16
    ) -> torch.Tensor:
        """Allocate a persistent tensor that won't be returned to pool"""
        if name in self.allocated_tensors:
            return self.allocated_tensors[name]

        tensor = self.pool.allocate(size_bytes, dtype)
        self.allocated_tensors[name] = tensor
        return tensor

    def allocate_temporary(
        self, size_bytes: int, dtype: torch.dtype = torch.float16
    ) -> torch.Tensor:
        """Allocate a temporary tensor from pool"""
        return self.pool.allocate(size_bytes, dtype)

    def free_temporary(self, tensor: torch.Tensor):
        """Free a temporary tensor back to pool"""
        self.pool.free(tensor)

    def free_persistent(self, name: str):
        """Free a persistent tensor"""
        if name in self.allocated_tensors:
            # Don't return persistent tensors to pool
            del self.allocated_tensors[name]

    def get_memory_info(self) -> Dict[str, Any]:
        """Get detailed memory information"""
        if not self.enabled:
            return {"gpu_available": False}

        try:
            allocated = torch.cuda.memory_allocated(self.device) / (1024**3)
            reserved = torch.cuda.memory_reserved(self.device) / (1024**3)
            total = torch.cuda.get_device_properties(self.device).total_memory / (
                1024**3
            )

            return {
                "gpu_available": True,
                "allocated_gb": allocated,
                "reserved_gb": reserved,
                "total_gb": total,
                "utilization": allocated / total,
                "pool_stats": self.pool.get_stats().__dict__,
                "pool_hit_rate": self.pool.get_cache_hit_rate(),
                "num_persistent_tensors": len(self.allocated_tensors),
            }
        except Exception as e:
            return {"gpu_available": False, "error": str(e)}

    def defragment(self):
        """Defragment memory pool (simplified)"""
        # In a real implementation, this would be more sophisticated
        self.pool.clear_pool()
        torch.cuda.empty_cache()

    def __del__(self):
        """Cleanup on destruction"""
        self.pool.clear_pool()
        self.allocated_tensors.clear()
