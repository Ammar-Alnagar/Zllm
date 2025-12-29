#!/usr/bin/env python3

"""
Simple test to check Mini-vLLM components
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def test_imports():
    """Test if all imports work"""
    print("Testing imports...")

    try:
        from config import ModelConfig, InferenceConfig

        print("✓ Config imports work")

        from transformers import AutoTokenizer

        print("✓ Transformers import works")

        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
        print("✓ Tokenizer loads successfully")

        # Test tokenization
        text = "Hello world"
        tokens = tokenizer.encode(text)
        decoded = tokenizer.decode(tokens)
        print(f"✓ Tokenization works: '{text}' -> {tokens} -> '{decoded}'")

        return True

    except Exception as e:
        print(f"❌ Import test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_config():
    """Test configuration creation"""
    print("\nTesting configuration...")

    try:
        from config import ModelConfig, InferenceConfig

        model_config = ModelConfig(
            hidden_size=1024,
            intermediate_size=3072,
            num_hidden_layers=28,
            num_attention_heads=16,
            num_key_value_heads=8,
            head_dim=128,
            vocab_size=151936,
            max_position_embeddings=40960,
            rope_theta=1000000.0,
            rms_norm_eps=1e-6,
        )
        print("✓ Model config created")

        inference_config = InferenceConfig(
            max_batch_size=1, max_seq_len=4096, dtype="fp16"
        )
        print("✓ Inference config created")

        print(f"  - Model has {model_config.num_hidden_layers} layers")
        print(f"  - Hidden size: {model_config.hidden_size}")
        print(f"  - Attention heads: {model_config.num_attention_heads}")

        return model_config, inference_config

    except Exception as e:
        print(f"❌ Config test failed: {e}")
        import traceback

        traceback.print_exc()
        return None, None


def test_model_runner_init(model_config, inference_config):
    """Test model runner initialization"""
    print("\nTesting model runner initialization...")

    try:
        from model_runner import ModelRunner

        model_runner = ModelRunner(model_config, inference_config)
        print("✓ Model runner created")

        return model_runner

    except Exception as e:
        print(f"❌ Model runner init failed: {e}")
        import traceback

        traceback.print_exc()
        return None


if __name__ == "__main__":
    print("🧪 Mini-vLLM Component Tests")
    print("=" * 40)

    # Test imports
    if not test_imports():
        sys.exit(1)

    # Test config
    model_config, inference_config = test_config()
    if model_config is None:
        sys.exit(1)

    # Test model runner
    model_runner = test_model_runner_init(model_config, inference_config)
    if model_runner is None:
        sys.exit(1)

    print("\n🎉 All component tests passed!")
    print("✅ Ready for inference testing")
