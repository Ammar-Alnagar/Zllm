# config.py - Configuration classes for Mini-vLLM

from typing import Optional
from dataclasses import dataclass
import torch


@dataclass
class ModelConfig:
    """Configuration for a transformer model (matches Qwen3 architecture)"""

    # Model dimensions
    hidden_size: int = 4096  # Hidden dimension (d_model)
    intermediate_size: int = 11008  # FFN intermediate dimension
    num_hidden_layers: int = 32  # Number of transformer layers
    num_attention_heads: int = 32  # Number of query heads
    num_key_value_heads: int = 8  # Number of KV heads (GQA)
    head_dim: int = 128  # Dimension per head
    vocab_size: int = 152064  # Vocabulary size (~150K)
    max_position_embeddings: int = 8192  # Maximum sequence length

    # RoPE configuration
    rope_theta: float = 1000000.0  # RoPE base frequency

    # Normalization
    rms_norm_eps: float = 1e-6  # RMSNorm epsilon

    # Data type
    dtype: torch.dtype = torch.float16

    # KV cache block size (for compatibility)
    block_size: int = 16

    @property
    def kv_head_ratio(self) -> int:
        """Ratio of query heads to KV heads (GQA)"""
        return self.num_attention_heads // self.num_key_value_heads

    @property
    def kv_cache_size_per_token(self) -> int:
        """Size of KV cache per token in bytes"""
        # K + V for each layer and each KV head
        element_size = torch.finfo(self.dtype).bits // 8
        return (
            2
            * self.num_hidden_layers
            * self.num_key_value_heads
            * self.head_dim
            * element_size
        )

    @property
    def num_layers(self) -> int:
        """Alias for num_hidden_layers for compatibility"""
        return self.num_hidden_layers

    @property
    def hidden_dim(self) -> int:
        """Alias for hidden_size for compatibility"""
        return self.hidden_size

    @property
    def num_heads(self) -> int:
        """Alias for num_attention_heads for compatibility"""
        return self.num_attention_heads

    @property
    def num_kv_heads(self) -> int:
        """Alias for num_key_value_heads for compatibility"""
        return self.num_key_value_heads

    @property
    def intermediate_dim(self) -> int:
        """Alias for intermediate_size for compatibility"""
        return self.intermediate_size

    @property
    def max_seq_len(self) -> int:
        """Alias for max_position_embeddings for compatibility"""
        return self.max_position_embeddings


@dataclass
class SamplingParams:
    """Parameters for token sampling during generation"""

    temperature: float = 1.0  # Temperature for softmax
    top_p: float = 1.0  # Top-p (nucleus) sampling threshold
    top_k: int = -1  # Top-k sampling (-1 = disabled)
    repetition_penalty: float = 1.0  # Penalty for repeated tokens
    max_tokens: int = 256  # Maximum tokens to generate
    stop_token_ids: Optional[list[int]] = None  # Stop generation on these tokens

    def __post_init__(self):
        if self.stop_token_ids is None:
            self.stop_token_ids = []

    @property
    def use_top_k(self) -> bool:
        return self.top_k > 0

    @property
    def use_top_p(self) -> bool:
        return self.top_p < 1.0


@dataclass
class InferenceConfig:
    """Configuration for inference engine"""

    max_batch_size: int = 32  # Maximum batch size
    max_seq_len: int = 4096  # Maximum sequence length
    block_size: int = 16  # KV cache block size
    dtype: str = "fp16"  # Data type ("fp16", "fp32")
    device: str = "cuda"  # Device to run on
    enable_cuda_graphs: bool = False  # Use CUDA graphs for optimization

    @property
    def torch_dtype(self) -> torch.dtype:
        """Convert string dtype to torch dtype"""
        if self.dtype == "fp16":
            return torch.float16
        elif self.dtype == "fp32":
            return torch.float32
        else:
            raise ValueError(f"Unsupported dtype: {self.dtype}")
