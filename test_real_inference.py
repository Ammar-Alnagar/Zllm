#!/usr/bin/env python3

"""
Test real Mini-vLLM inference with Qwen/Qwen3-0.6B
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model_runner import ModelRunner, GenerationRequest
from config import ModelConfig, InferenceConfig
from transformers import AutoTokenizer


def test_real_inference():
    """Test actual Mini-vLLM inference"""
    print("🧪 Testing Real Mini-vLLM Inference with Qwen/Qwen3-0.6B")
    print("=" * 60)

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

        inference_config = InferenceConfig(
            max_batch_size=1, max_seq_len=4096, dtype="fp16"
        )

        # Initialize model runner
        print("Initializing model runner...")
        model_runner = ModelRunner(model_config, inference_config)
        model_runner.set_tokenizer(tokenizer)
        print("✓ Model runner initialized")

        # Test tokenization
        test_prompt = "Hello, how are you today?"
        tokens = tokenizer.encode(test_prompt)
        print(f"✓ Tokenization test: '{test_prompt}' -> {len(tokens)} tokens")

        # Test inference
        print("\n🤖 Testing actual inference...")
        gen_request = GenerationRequest(
            request_id="test_inference",
            prompt=test_prompt,
            max_new_tokens=20,
            temperature=0.8,
            top_p=0.9,
            top_k=50,
        )

        # Add and process request
        print("Adding request to model runner...")
        model_runner.add_request(gen_request)

        print("Processing inference...")
        result = model_runner.process_request("test_inference")

        print("\n🎉 Inference completed!")
        print(f"📝 Original prompt: '{test_prompt}'")
        print(f"🤖 Generated text: '{result.generated_text}'")
        print(f"📊 Tokens generated: {len(result.generated_tokens)}")
        print(f"🏁 Finish reason: {result.finish_reason}")

        # Decode tokens to verify
        if result.generated_tokens:
            decoded = tokenizer.decode(
                result.generated_tokens, skip_special_tokens=True
            )
            print(f"🔍 Decoded tokens: '{decoded}'")

        return True

    except Exception as e:
        print(f"❌ Inference test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_real_inference()
    if success:
        print("\n🎊 Real Mini-vLLM inference test passed!")
        print("✅ Qwen/Qwen3-0.6B model is working with actual inference!")
    else:
        print("\n💥 Real inference test failed!")
        sys.exit(1)
