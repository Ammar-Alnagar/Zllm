#!/usr/bin/env python3

"""
Interactive chat with Mini-vLLM and Qwen/Qwen3-0.6B
"""

import sys
import os
import torch
from transformers import AutoTokenizer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "zllm", "python"))


def main():
    print("🤖 Mini-vLLM Chat Demo with Qwen/Qwen3-0.6B")
    print("=" * 60)

    # Load model configuration
    print("📋 Loading Qwen model configuration...")
    from transformers import AutoConfig

    model_name = "Qwen/Qwen3-0.6B"
    try:
        config = AutoConfig.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        print("✅ Model loaded successfully!")
        print(f"   • Architecture: {config.model_type}")
        print(f"   • Hidden size: {config.hidden_size}")
        print(f"   • Layers: {config.num_hidden_layers}")
        print(f"   • Attention heads: {config.num_attention_heads}")
        print(f"   • Vocabulary size: {config.vocab_size}")
        print(f"   • CUDA available: {torch.cuda.is_available()}")

        if torch.cuda.is_available():
            print(f"   • GPU: {torch.cuda.get_device_name()}")

        print("\n💬 Ready for interactive chat!")
        print("Type 'quit' or 'exit' to stop.")
        print("-" * 40)

        while True:
            try:
                # Get user input
                user_input = input("\n👤 You: ").strip()

                if user_input.lower() in ["quit", "exit", "bye"]:
                    print("\n👋 Goodbye!")
                    break

                if not user_input:
                    continue

                # Tokenize input
                tokens = tokenizer.encode(user_input)
                print(f"📝 Tokenized: {len(tokens)} tokens")

                # Show some token details
                if len(tokens) <= 10:
                    print(f"   Tokens: {tokens}")
                else:
                    print(f"   Tokens: {tokens[:5]} ... {tokens[-5:]}")

                # Decode back
                decoded = tokenizer.decode(tokens)
                print(f"   Decoded: '{decoded}'")

                # Simulate model response (since we don't have weights loaded)
                print("\n🤖 Mini-vLLM Response:")
                print("   (Model weights not loaded - this is a demonstration)")
                print("   In a full implementation, this would generate a response!")
                print("   🚀 Mini-vLLM supports:")
                print(
                    "      • Custom CUDA kernels (RMSNorm, RoPE, SwiGLU, Flash Attention)"
                )
                print("      • Paged KV cache with 16-token blocks")
                print("      • Continuous batching scheduler")
                print("      • FastAPI server with OpenAI-compatible API")

            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
                continue

    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        print("\n💡 Make sure you have:")
        print("   • transformers library installed")
        print("   • Access to download Qwen/Qwen3-0.6B")
        print("   • Internet connection for model download")


if __name__ == "__main__":
    main()
