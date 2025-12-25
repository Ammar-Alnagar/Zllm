# Phase 3: Model Runner and Tokenizer Integration

## Table of Contents

1. [Model Runner Overview](#model-runner-overview)
2. [Weight Loading](#weight-loading)
3. [Tokenizer Integration](#tokenizer-integration)
4. [Sampling Strategies](#sampling-strategies)
5. [Complete Model Runner](#complete-model-runner)
6. [Testing](#testing)

---

## Model Runner Overview

The **Model Runner** coordinates model execution, handling weight loading, forward passes, and token sampling.

```
                    Model Runner Architecture

┌─────────────────────────────────────────────────────────┐
│                     Model Runner                        │
│                                                         │
│  ┌─────────────────────────────────────────────────┐   │
│  │              Weight Manager                      │   │
│  │  Load from: HuggingFace, SafeTensors, GGUF      │   │
│  └────────────────────┬────────────────────────────┘   │
│                       │                                 │
│  ┌────────────────────▼────────────────────────────┐   │
│  │              Forward Engine                      │   │
│  │  Embedding → Layers → RMSNorm → LM Head         │   │
│  └────────────────────┬────────────────────────────┘   │
│                       │                                 │
│  ┌────────────────────▼────────────────────────────┐   │
│  │              Sampler                             │   │
│  │  Temperature → Top-K → Top-P → Sample           │   │
│  └─────────────────────────────────────────────────┘   │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

---

## Weight Loading

Create file: `mini_vllm/python/mini_vllm/model_loader.py`

```python
"""Model Weight Loading - Load weights from HuggingFace/SafeTensors"""

from dataclasses import dataclass
from typing import Dict, List
from pathlib import Path
import torch
import json


@dataclass
class ModelConfig:
    """Configuration for Qwen3-like model"""
    vocab_size: int = 151936
    hidden_size: int = 4096
    intermediate_size: int = 11008
    num_hidden_layers: int = 32
    num_attention_heads: int = 32
    num_key_value_heads: int = 8
    head_dim: int = 128
    max_position_embeddings: int = 32768
    rms_norm_eps: float = 1e-6
    rope_theta: float = 1000000.0

    @classmethod
    def from_pretrained(cls, model_path: str) -> "ModelConfig":
        """Load config from HuggingFace format"""
        config_path = Path(model_path) / "config.json"
        with open(config_path) as f:
            d = json.load(f)
        return cls(
            vocab_size=d.get("vocab_size", 151936),
            hidden_size=d.get("hidden_size", 4096),
            intermediate_size=d.get("intermediate_size", 11008),
            num_hidden_layers=d.get("num_hidden_layers", 32),
            num_attention_heads=d.get("num_attention_heads", 32),
            num_key_value_heads=d.get("num_key_value_heads", 8),
            head_dim=d.get("head_dim", 128),
            max_position_embeddings=d.get("max_position_embeddings", 32768),
            rms_norm_eps=d.get("rms_norm_eps", 1e-6),
            rope_theta=d.get("rope_theta", 1000000.0),
        )


class WeightLoader:
    """Load model weights from various formats."""

    def __init__(self, model_path: str, dtype: torch.dtype = torch.float16):
        self.model_path = Path(model_path)
        self.dtype = dtype
        self.config = ModelConfig.from_pretrained(model_path)

    def load_weights(self, device: str = "cuda") -> Dict[str, torch.Tensor]:
        """Load all model weights"""
        weights = {}

        # Try SafeTensors first
        safetensor_files = list(self.model_path.glob("*.safetensors"))
        if safetensor_files:
            from safetensors import safe_open
            for f in sorted(safetensor_files):
                with safe_open(f, framework="pt", device=device) as st:
                    for key in st.keys():
                        weights[key] = st.get_tensor(key).to(self.dtype)
            return weights

        # Fall back to PyTorch
        for f in sorted(self.model_path.glob("*.bin")):
            state = torch.load(f, map_location=device)
            weights.update({k: v.to(self.dtype) for k, v in state.items()})

        return weights


class LayerWeights:
    """Container for one transformer layer's weights"""

    def __init__(self, layer_idx: int, weights: Dict[str, torch.Tensor]):
        self.layer_idx = layer_idx

        # Attention weights
        self.q_proj = weights.get("self_attn.q_proj.weight")
        self.k_proj = weights.get("self_attn.k_proj.weight")
        self.v_proj = weights.get("self_attn.v_proj.weight")
        self.o_proj = weights.get("self_attn.o_proj.weight")

        # FFN weights
        self.gate_proj = weights.get("mlp.gate_proj.weight")
        self.up_proj = weights.get("mlp.up_proj.weight")
        self.down_proj = weights.get("mlp.down_proj.weight")

        # Norms
        self.input_layernorm = weights.get("input_layernorm.weight")
        self.post_attention_layernorm = weights.get("post_attention_layernorm.weight")


class ModelWeights:
    """Container for all model weights"""

    def __init__(self, weights: Dict[str, torch.Tensor], config: ModelConfig):
        self.config = config

        # Embedding
        self.embed_tokens = weights.get("model.embed_tokens.weight")

        # Final norm
        self.final_norm = weights.get("model.norm.weight")

        # LM head (may be tied to embed_tokens)
        self.lm_head = weights.get("lm_head.weight", self.embed_tokens)

        # Layer weights
        self.layers: List[LayerWeights] = []
        for i in range(config.num_hidden_layers):
            layer_weights = {
                k.replace(f"model.layers.{i}.", ""): v
                for k, v in weights.items()
                if k.startswith(f"model.layers.{i}.")
            }
            self.layers.append(LayerWeights(i, layer_weights))
```

---

## Tokenizer Integration

Create file: `mini_vllm/python/mini_vllm/tokenizer.py`

```python
"""Tokenizer Integration using tiktoken"""

from typing import List
import tiktoken


class Tokenizer:
    """Tokenizer for Qwen3 using tiktoken."""

    # Special token IDs (adjust for your model)
    PAD_ID = 0
    BOS_ID = 1
    EOS_ID = 2

    def __init__(self, model_name: str = "cl100k_base"):
        """Initialize with tiktoken encoding."""
        self.encoding = tiktoken.get_encoding(model_name)
        self.vocab_size = self.encoding.n_vocab

    def encode(
        self,
        text: str,
        add_bos: bool = True,
        add_eos: bool = False
    ) -> List[int]:
        """Encode text to token IDs."""
        tokens = self.encoding.encode(text)
        if add_bos:
            tokens = [self.BOS_ID] + tokens
        if add_eos:
            tokens = tokens + [self.EOS_ID]
        return tokens

    def decode(self, tokens: List[int], skip_special: bool = True) -> str:
        """Decode token IDs to text."""
        if skip_special:
            tokens = [t for t in tokens
                     if t not in (self.PAD_ID, self.BOS_ID, self.EOS_ID)]
        return self.encoding.decode(tokens)

    def batch_encode(self, texts: List[str], add_bos: bool = True) -> List[List[int]]:
        """Encode multiple texts."""
        return [self.encode(t, add_bos=add_bos) for t in texts]
```

---

## Sampling Strategies

Create file: `mini_vllm/python/mini_vllm/sampling.py`

```python
"""Token Sampling Strategies"""

from dataclasses import dataclass, field
from typing import List
import torch
import torch.nn.functional as F


@dataclass
class SamplingParams:
    """Sampling parameters"""
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = -1
    max_tokens: int = 256
    repetition_penalty: float = 1.0
    stop_token_ids: List[int] = field(default_factory=lambda: [2])


class Sampler:
    """Token sampler with various strategies."""

    def __init__(self, vocab_size: int):
        self.vocab_size = vocab_size

    def sample(
        self,
        logits: torch.Tensor,  # [batch, vocab_size]
        params: SamplingParams
    ) -> torch.Tensor:
        """Sample next tokens from logits."""
        # Apply temperature
        if params.temperature > 0:
            logits = logits / params.temperature
        else:
            # Greedy decoding
            return logits.argmax(dim=-1)

        # Apply top-k
        if params.top_k > 0:
            logits = self._top_k_filter(logits, params.top_k)

        # Apply top-p (nucleus sampling)
        if params.top_p < 1.0:
            logits = self._top_p_filter(logits, params.top_p)

        # Sample from distribution
        probs = F.softmax(logits, dim=-1)
        return torch.multinomial(probs, num_samples=1).squeeze(-1)

    def _top_k_filter(self, logits: torch.Tensor, k: int) -> torch.Tensor:
        """Keep only top-k logits."""
        values, _ = torch.topk(logits, k, dim=-1)
        min_values = values[:, -1].unsqueeze(-1)
        return torch.where(
            logits < min_values,
            torch.full_like(logits, float('-inf')),
            logits
        )

    def _top_p_filter(self, logits: torch.Tensor, p: float) -> torch.Tensor:
        """Keep tokens with cumulative probability <= p."""
        sorted_logits, sorted_idx = torch.sort(logits, descending=True)
        probs = F.softmax(sorted_logits, dim=-1)
        cumsum = torch.cumsum(probs, dim=-1)

        # Find cutoff
        mask = cumsum > p
        mask[:, 1:] = mask[:, :-1].clone()
        mask[:, 0] = False

        sorted_logits[mask] = float('-inf')

        # Unsort
        result = torch.zeros_like(logits)
        result.scatter_(-1, sorted_idx, sorted_logits)
        return result
```

---

## Complete Model Runner

Create file: `mini_vllm/python/mini_vllm/model_runner.py`

```python
"""Complete Model Runner - Orchestrates inference"""

from typing import Optional
import torch
import torch.nn.functional as F

from .model_loader import ModelConfig, WeightLoader, ModelWeights
from .tokenizer import Tokenizer
from .sampling import Sampler, SamplingParams
from .kv_cache import KVCacheManager, CacheConfig


class ModelRunner:
    """Orchestrates model inference."""

    def __init__(
        self,
        model_path: str,
        dtype: torch.dtype = torch.float16,
        device: str = "cuda"
    ):
        self.device = device
        self.dtype = dtype

        # Load model
        loader = WeightLoader(model_path, dtype)
        self.config = loader.config
        weights_dict = loader.load_weights(device)
        self.weights = ModelWeights(weights_dict, self.config)

        # Initialize tokenizer
        self.tokenizer = Tokenizer()

        # Initialize sampler
        self.sampler = Sampler(self.config.vocab_size)

        # Initialize KV cache
        cache_config = CacheConfig(
            num_blocks=1000,
            block_size=16,
            num_kv_heads=self.config.num_key_value_heads,
            head_dim=self.config.head_dim,
            dtype=dtype,
            device=device
        )
        self.kv_cache = KVCacheManager(cache_config)

        # Precompute RoPE
        self._init_rope()

    def _init_rope(self):
        """Precompute RoPE cos/sin tables."""
        dim = self.config.head_dim
        max_seq = self.config.max_position_embeddings
        theta = self.config.rope_theta

        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        positions = torch.arange(max_seq).float()
        freqs = torch.outer(positions, inv_freq)

        self.rope_cos = freqs.cos().to(self.device)
        self.rope_sin = freqs.sin().to(self.device)

    def generate(
        self,
        prompt: str,
        sampling_params: Optional[SamplingParams] = None
    ) -> str:
        """Generate text from prompt."""
        if sampling_params is None:
            sampling_params = SamplingParams()

        # Encode prompt
        input_ids = self.tokenizer.encode(prompt)
        input_tensor = torch.tensor([input_ids], device=self.device)

        # Allocate KV cache
        seq_id = 0
        self.kv_cache.allocate_sequence(seq_id, len(input_ids))

        # Prefill
        positions = torch.arange(len(input_ids), device=self.device).unsqueeze(0)
        logits = self._forward(input_tensor, positions)

        # Sample first token
        next_token = self.sampler.sample(logits[:, -1], sampling_params)
        output_tokens = [next_token.item()]

        # Decode loop
        for _ in range(sampling_params.max_tokens - 1):
            # Check stop conditions
            if output_tokens[-1] in sampling_params.stop_token_ids:
                break

            # Extend cache
            self.kv_cache.extend_sequence(seq_id, 1)

            # Forward one token
            input_tensor = next_token.unsqueeze(0).unsqueeze(0)
            pos = len(input_ids) + len(output_tokens) - 1
            positions = torch.tensor([[pos]], device=self.device)

            logits = self._forward(input_tensor, positions)
            next_token = self.sampler.sample(logits[:, -1], sampling_params)
            output_tokens.append(next_token.item())

        # Cleanup
        self.kv_cache.free_sequence(seq_id)

        # Decode
        return self.tokenizer.decode(output_tokens)

    def _forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass through model."""
        # Embedding
        hidden = F.embedding(input_ids, self.weights.embed_tokens)

        # Layers
        for layer in self.weights.layers:
            hidden = self._forward_layer(hidden, positions, layer)

        # Final norm
        hidden = self._rmsnorm(hidden, self.weights.final_norm)

        # LM head
        logits = F.linear(hidden, self.weights.lm_head)

        return logits

    def _forward_layer(self, hidden, positions, layer):
        """Forward through one transformer layer."""
        residual = hidden

        # Pre-attention norm
        hidden = self._rmsnorm(hidden, layer.input_layernorm)

        # Attention
        hidden = self._attention(hidden, positions, layer)
        hidden = residual + hidden
        residual = hidden

        # Pre-FFN norm
        hidden = self._rmsnorm(hidden, layer.post_attention_layernorm)

        # FFN (SwiGLU)
        gate = F.linear(hidden, layer.gate_proj)
        up = F.linear(hidden, layer.up_proj)
        hidden = F.silu(gate) * up
        hidden = F.linear(hidden, layer.down_proj)

        hidden = residual + hidden
        return hidden

    def _rmsnorm(self, x, weight):
        """RMSNorm."""
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.config.rms_norm_eps)
        return weight * x

    def _attention(self, hidden, positions, layer):
        """Self-attention."""
        batch, seq_len, _ = hidden.shape

        # Projections
        q = F.linear(hidden, layer.q_proj)
        k = F.linear(hidden, layer.k_proj)
        v = F.linear(hidden, layer.v_proj)

        # Reshape
        num_heads = self.config.num_attention_heads
        num_kv = self.config.num_key_value_heads
        head_dim = self.config.head_dim

        q = q.view(batch, seq_len, num_heads, head_dim)
        k = k.view(batch, seq_len, num_kv, head_dim)
        v = v.view(batch, seq_len, num_kv, head_dim)

        # Apply RoPE
        q, k = self._apply_rope(q, k, positions)

        # Expand KV for GQA
        if num_kv != num_heads:
            repeat = num_heads // num_kv
            k = k.repeat_interleave(repeat, dim=2)
            v = v.repeat_interleave(repeat, dim=2)

        # Attention computation
        q = q.transpose(1, 2)  # [batch, heads, seq, dim]
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / (head_dim ** 0.5)

        # Causal mask
        mask = torch.triu(torch.ones(seq_len, seq_len, device=q.device), 1).bool()
        scores.masked_fill_(mask, float('-inf'))

        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, v)

        out = out.transpose(1, 2).reshape(batch, seq_len, -1)
        return F.linear(out, layer.o_proj)

    def _apply_rope(self, q, k, positions):
        """Apply rotary embeddings."""
        cos = self.rope_cos[positions].unsqueeze(2)
        sin = self.rope_sin[positions].unsqueeze(2)

        def rotate_half(x):
            x1, x2 = x.chunk(2, dim=-1)
            return torch.cat([-x2, x1], dim=-1)

        q = q * cos + rotate_half(q) * sin
        k = k * cos + rotate_half(k) * sin

        return q, k
```

---

## Testing

Create file: `mini_vllm/tests/python/test_model_runner.py`

```python
"""Test Model Runner Components"""

import pytest
import torch
from mini_vllm.sampling import Sampler, SamplingParams
from mini_vllm.tokenizer import Tokenizer


class TestTokenizer:
    def test_encode_decode(self):
        tok = Tokenizer()
        text = "Hello, world!"
        tokens = tok.encode(text)
        assert len(tokens) > 1  # At least BOS + text
        decoded = tok.decode(tokens[1:])  # Skip BOS
        assert len(decoded) > 0


class TestSampler:
    def test_greedy(self):
        sampler = Sampler(1000)
        logits = torch.randn(2, 1000)
        params = SamplingParams(temperature=0)
        tokens = sampler.sample(logits, params)
        assert torch.equal(tokens, logits.argmax(dim=-1))

    def test_top_k(self):
        sampler = Sampler(1000)
        logits = torch.randn(1, 1000)
        params = SamplingParams(temperature=1.0, top_k=10)
        for _ in range(10):
            token = sampler.sample(logits, params)
            assert 0 <= token.item() < 1000

    def test_top_p(self):
        sampler = Sampler(1000)
        logits = torch.randn(1, 1000)
        params = SamplingParams(temperature=1.0, top_p=0.9)
        token = sampler.sample(logits, params)
        assert 0 <= token.item() < 1000


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
```

---

## Summary

| Component        | Purpose                                |
| ---------------- | -------------------------------------- |
| **WeightLoader** | Load model weights from HF/SafeTensors |
| **ModelConfig**  | Model architecture configuration       |
| **Tokenizer**    | Encode/decode text using tiktoken      |
| **Sampler**      | Temperature, top-k, top-p sampling     |
| **ModelRunner**  | Orchestrate end-to-end inference       |

---

## What's Next

Now we'll build the **Inference Engine** that combines scheduler and model runner.

Continue to: [13_inference_engine.md](./13_inference_engine.md)
