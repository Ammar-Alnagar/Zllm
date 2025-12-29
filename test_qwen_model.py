#!/usr/bin/env python3

"""
Test script to load Qwen/Qwen3-0.6B model and inspect its configuration
"""

import torch
from transformers import AutoTokenizer, AutoConfig


def test_model_loading():
    """Test loading the Qwen model"""
    print("=== Testing Qwen Model Loading ===")

    model_name = "Qwen/Qwen3-0.6B"

    try:
        # Load with transformers first
        print(f"Loading {model_name} with transformers...")
        config = AutoConfig.from_pretrained(model_name)
        print(f"✓ Model config loaded: {config.model_type}")
        print(f"  - Hidden size: {config.hidden_size}")
        print(f"  - Num layers: {config.num_hidden_layers}")
        print(f"  - Num heads: {config.num_attention_heads}")
        print(f"  - Vocab size: {config.vocab_size}")
        print(f"  - Max position embeddings: {config.max_position_embeddings}")

        # Check Qwen-specific attributes
        if hasattr(config, "intermediate_size"):
            print(f"  - Intermediate size: {config.intermediate_size}")
        if hasattr(config, "num_key_value_heads"):
            print(f"  - Num KV heads: {config.num_key_value_heads}")
        if hasattr(config, "head_dim"):
            print(f"  - Head dim: {config.head_dim}")
        if hasattr(config, "rope_theta"):
            print(f"  - RoPE theta: {config.rope_theta}")
        if hasattr(config, "rms_norm_eps"):
            print(f"  - RMS norm epsilon: {config.rms_norm_eps}")

        # Try to load tokenizer
        print("\nLoading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        print(f"✓ Tokenizer loaded: {type(tokenizer).__name__}")

        # Test tokenization
        test_text = "Hello, how are you?"
        tokens = tokenizer.encode(test_text)
        print(f"  - Test text: '{test_text}'")
        print(f"  - Tokenized: {tokens}")
        print(f"  - Decoded: '{tokenizer.decode(tokens)}'")

        # Create Mini-vLLM style config manually
        print("\nCreating Mini-vLLM style configuration...")
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

        print("✓ Mini-vLLM configuration created:")
        for key, value in minivllm_config.items():
            print(f"  - {key}: {value}")

        # Test memory estimation
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
        print(f"  - Estimated memory (FP16): {memory_gb_fp16:.1f} GB")
        return True

    except Exception as e:
        print(f"✗ Failed to load model: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_model_loading()
    if success:
        print("\n🎉 Model loading test passed!")
    else:
        print("\n❌ Model loading test failed!")
        exit(1)
