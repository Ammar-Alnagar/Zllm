#!/usr/bin/env python3

"""
Debug test to see what's happening in the attention computation
"""

import sys
import os
import numpy as np

# Add the zllm/python directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'zllm', 'python'))

def debug_attention():
    """Debug attention computation"""
    print("Debugging attention computation...")
    
    try:
        import engine
        
        # Create small config
        model_config = engine.ModelConfig(
            num_layers=1,
            num_heads=2,
            num_kv_heads=1,
            head_dim=16,
            hidden_dim=32,
            intermediate_dim=64,
            vocab_size=10,
            max_seq_len=32,
            dtype="fp32"
        )
        
        # Create engine
        engine_instance = engine.InferenceEngine(model_config, engine.InferenceConfig())
        
        # Mock tokenizer
        class MockTokenizer:
            def encode(self, text):
                return [1, 2, 3]
            
            @property
            def eos_token_id(self):
                return 0
        
        # Add sequence
        seq_id = engine_instance.add_sequence("Test", MockTokenizer())
        
        # Test prefill
        print("Testing prefill...")
        logits = engine_instance.prefill(seq_id)
        print(f"Prefill logits shape: {logits.shape}")
        
        # Test decode with first token
        print("Testing decode with first token...")
        next_token = 4
        logits = engine_instance.decode(seq_id, next_token)
        print(f"Decode logits shape: {logits.shape}")
        
        print("✓ Debug test passed")
        return True
    except Exception as e:
        print(f"✗ Debug test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    debug_attention()