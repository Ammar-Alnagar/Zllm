#!/usr/bin/env python3

"""
Test script to check tokenizer and model runner initialization
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model_runner import ModelRunner
from config import ModelConfig, InferenceConfig
from transformers import AutoTokenizer


def test_initialization():
    """Test the initialization process"""
    print("Testing Mini-vLLM initialization...")

    try:
        # Load tokenizer
        print("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
        print("✓ Tokenizer loaded")

        # Create model config
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

        # Create inference config
        inference_config = InferenceConfig(
            max_batch_size=1, max_seq_len=4096, dtype="fp16"
        )
        print("✓ Inference config created")

        # Initialize model runner
        model_runner = ModelRunner(model_config, inference_config)
        model_runner.set_tokenizer(tokenizer)
        print("✓ Model runner initialized with tokenizer")

        # Test tokenization
        test_prompt = "Hello, world!"
        tokens = tokenizer.encode(test_prompt)
        print(f"✓ Tokenization test: '{test_prompt}' -> {len(tokens)} tokens")

        print("\n🎉 All initialization tests passed!")
        return True

    except Exception as e:
        print(f"❌ Initialization failed: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    test_initialization()
