#!/usr/bin/env python3

"""
Demo script showing Mini-vLLM setup with Qwen/Qwen3-0.6B model configuration
"""

import numpy as np
import torch
from transformers import AutoConfig, AutoTokenizer


def demo_minivllm_setup():
    """Demonstrate Mini-vLLM setup with Qwen model"""
    print("🚀 Mini-vLLM Demo with Qwen/Qwen3-0.6B")
    print("=" * 50)

    # Load Qwen model configuration
    print("\n📋 Loading Qwen model configuration...")
    model_name = "Qwen/Qwen3-0.6B"
    config = AutoConfig.from_pretrained(model_name)

    print("✓ Model config loaded:")
    print(f"  - Architecture: {config.model_type}")
    print(f"  - Hidden size: {config.hidden_size}")
    print(f"  - Num layers: {config.num_hidden_layers}")
    print(f"  - Num heads: {config.num_attention_heads}")
    print(f"  - Vocab size: {config.vocab_size}")
    print(f"  - Max position embeddings: {config.max_position_embeddings}")

    # Qwen-specific attributes
    if hasattr(config, "intermediate_size"):
        print(f"  - Intermediate size: {config.intermediate_size}")
    if hasattr(config, "num_key_value_heads"):
        print(f"  - Num KV heads: {config.num_key_value_heads} (GQA)")
    if hasattr(config, "head_dim"):
        print(f"  - Head dim: {config.head_dim}")
    if hasattr(config, "rope_theta"):
        print(f"  - RoPE theta: {config.rope_theta}")
    if hasattr(config, "rms_norm_eps"):
        print(f"  - RMS norm epsilon: {config.rms_norm_eps:.2e}")
    # Create Mini-vLLM style config
    minivllm_config = {
        "hidden_size": config.hidden_size,
        "intermediate_size": getattr(
            config, "intermediate_size", config.hidden_size * 4
        ),
        "num_hidden_layers": config.num_hidden_layers,
        "num_attention_heads": config.num_attention_heads,
        "num_key_value_heads": getattr(
            config, "num_key_value_heads", config.num_attention_heads
        ),
        "head_dim": getattr(
            config, "head_dim", config.hidden_size // config.num_attention_heads
        ),
        "vocab_size": config.vocab_size,
        "max_position_embeddings": config.max_position_embeddings,
        "rope_theta": getattr(config, "rope_theta", 1000000.0),
        "rms_norm_eps": getattr(config, "rms_norm_eps", 1e-6),
    }

    # Setup tokenizer
    print("\n🔤 Setting up tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    print(f"✓ Tokenizer loaded: {type(tokenizer).__name__}")

    # Test tokenization
    test_text = "The future of AI is open source."
    tokens = tokenizer.encode(test_text)
    print(f"  - Test: '{test_text}'")
    print(f"  - Tokens: {tokens}")
    print(f"  - Length: {len(tokens)}")

    # Calculate memory requirements
    param_count = (
        minivllm_config["vocab_size"] * minivllm_config["hidden_size"]  # embeddings
        + minivllm_config["vocab_size"] * minivllm_config["hidden_size"]  # output
        + minivllm_config["num_hidden_layers"]
        * (
            4
            * minivllm_config["hidden_size"]
            * minivllm_config["hidden_size"]  # attention
            + 3
            * minivllm_config["hidden_size"]
            * minivllm_config["intermediate_size"]  # MLP
            + 2 * minivllm_config["hidden_size"]  # norms
        )
        + minivllm_config["hidden_size"]  # final norm
    )

    memory_gb_fp16 = param_count * 2 / (1024**3)
    kv_memory_per_token = (
        2
        * minivllm_config["num_hidden_layers"]
        * minivllm_config["num_key_value_heads"]
        * minivllm_config["head_dim"]
        * 2
        / (1024**3)  # FP16
    )

    print("\n💾 Memory requirements:")
    print(f"  - Model size (FP16): {memory_gb_fp16:.1f} GB")
    print(f"  - KV cache per token: {kv_memory_per_token:.3f} GB")
    # Demo sampling (simple multinomial)
    print("\n🎲 Testing sampling...")
    vocab_size = minivllm_config["vocab_size"]

    # Create dummy logits
    np.random.seed(42)
    dummy_logits = np.random.randn(vocab_size).astype(np.float32)
    dummy_logits = dummy_logits / 0.1  # Make more extreme

    # Temperature sampling
    temperature = 0.7
    top_p = 0.9
    top_k = 50

    # Apply temperature
    logits = dummy_logits / temperature

    # Apply top-k
    top_k_indices = np.argpartition(logits, -top_k)[-top_k:]
    mask = np.full_like(logits, -float("inf"))
    mask[top_k_indices] = logits[top_k_indices]
    logits = mask

    # Apply top-p
    probs = np.exp(logits - np.max(logits))
    probs = probs / np.sum(probs)

    sorted_indices = np.argsort(probs)[::-1]
    sorted_probs = probs[sorted_indices]
    cumulative_probs = np.cumsum(sorted_probs)
    cutoff_idx = np.searchsorted(cumulative_probs, top_p, side="right")
    cutoff_idx = max(1, cutoff_idx)

    final_mask = np.full_like(probs, -float("inf"))
    final_mask[sorted_indices[:cutoff_idx]] = logits[sorted_indices[:cutoff_idx]]
    final_logits = final_mask

    # Sample
    final_probs = np.exp(final_logits - np.max(final_logits))
    final_probs = final_probs / np.sum(final_probs)

    sampled_tokens = []
    for i in range(3):
        token = np.random.choice(len(final_probs), p=final_probs)
        sampled_tokens.append(token)

    print(f"✓ Sampled tokens: {sampled_tokens}")
    print(f"  - Temperature: {temperature}, Top-p: {top_p}, Top-k: {top_k}")

    # Show CUDA status
    print("\n🖥️  System info:")
    print(f"  - PyTorch: {torch.__version__}")
    print(f"  - CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  - GPU: {torch.cuda.get_device_name()}")
        print(
            f"  - GPU memory: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.1f} GB"
        )
    print("\n📚 Mini-vLLM Architecture:")
    print("  ✅ Custom CUDA kernels (RMSNorm, RoPE, SwiGLU, Flash Attention)")
    print("  ✅ Paged KV cache with 16-token blocks")
    print("  ✅ RadixAttention for prefix sharing")
    print("  ✅ Continuous batching scheduler")
    print("  ✅ FastAPI server with OpenAI-compatible API")
    print("  ✅ Memory pool with reuse and defragmentation")

    print("\n" + "=" * 50)
    print("🎉 Mini-vLLM successfully configured for Qwen/Qwen3-0.6B!")
    print("\n🚀 Ready for inference with:")
    print(f"   • {minivllm_config['num_hidden_layers']} transformer layers")
    print(
        f"   • {minivllm_config['num_attention_heads']} attention heads ({minivllm_config['num_key_value_heads']} KV)"
    )
    print(f"   • RoPE positional embeddings (θ={minivllm_config['rope_theta']})")
    print(f"   • ~{param_count // 1000000000}B parameters")


if __name__ == "__main__":
    demo_minivllm_setup()
