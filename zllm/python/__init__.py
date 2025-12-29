"""
Mini-vLLM: Educational LLM Inference Engine
===========================================

A from-scratch implementation of vLLM concepts for learning purposes.

This package provides:
- Custom CUDA kernels for attention, normalization, and activations
- Paged KV cache with RadixAttention for prefix sharing
- Continuous batching scheduler
- FastAPI server for deployment

Example:
    >>> from mini_vllm import InferenceEngine, SamplingParams
    >>> engine = InferenceEngine(model_config, inference_config)
    >>> params = SamplingParams(temperature=0.7, max_tokens=100)
    >>> output = engine.generate(seq_id, 10, tokenizer)
    >>> print(output)
"""

__version__ = "0.1.0"
__author__ = "Your Name"

# Import main components
from .config import ModelConfig, SamplingParams
from .engine import InferenceEngine
from .tokenizer import Tokenizer

# Try to import CUDA extensions
try:
    from . import mini_vllm_ops as _ops

    CUDA_AVAILABLE = True
except ImportError:
    CUDA_AVAILABLE = False
    import warnings

    warnings.warn(
        "CUDA extensions not found. Install with: pip install -e .", ImportWarning
    )

__all__ = [
    "InferenceEngine",
    "ModelConfig",
    "SamplingParams",
    "Tokenizer",
    "CUDA_AVAILABLE",
]
