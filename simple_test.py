#!/usr/bin/env python3

"""
Simple test script to verify the Mini-vLLM implementation
"""

import sys
import os
import numpy as np

# Add the zllm/python directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'zllm', 'python'))

def test_engine_creation():
    """Test engine creation"""
    print("Testing engine creation...")
    
    try:
        # Import directly from the python directory
        import engine
        
        # Create small config for testing
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
        
        inference_config = engine.InferenceConfig(
            max_tokens=10,
            temperature=0.5,
            top_p=1.0,
            top_k=0
        )
        
        # Create engine
        engine_instance = engine.InferenceEngine(model_config, inference_config)
        print("✓ Engine created successfully")
        return True
    except Exception as e:
        print(f"✗ Failed to create engine: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_model_runner():
    """Test model runner"""
    print("\nTesting model runner...")
    
    try:
        # Import directly from the python directory
        import model_runner
        import engine
        
        # Create configs
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
        
        inference_config = engine.InferenceConfig(
            max_tokens=10,
            temperature=0.5,
            top_p=1.0,
            top_k=0
        )
        
        # Create runner
        runner = model_runner.ModelRunner(model_config, inference_config)
        
        # Mock tokenizer
        class MockTokenizer:
            def encode(self, text):
                return [1, 2, 3]
            
            def decode(self, tokens):
                return f"Result: {' '.join(str(t) for t in tokens)}"
            
            @property
            def eos_token_id(self):
                return 0
        
        runner.set_tokenizer(MockTokenizer())
        
        # Create and add request
        request = model_runner.GenerationRequest(
            request_id="test_001",
            prompt="Hello, world!",
            max_new_tokens=3,
            temperature=0.5
        )
        
        runner.add_request(request)
        print("✓ Model runner created and request added successfully")
        return True
    except Exception as e:
        print(f"✗ Failed to test model runner: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_basic_generation():
    """Test basic generation"""
    print("\nTesting basic generation...")
    
    try:
        # Import directly from the python directory
        import model_runner
        import engine
        
        # Create configs
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
        
        inference_config = engine.InferenceConfig(
            max_tokens=10,
            temperature=0.5,
            top_p=1.0,
            top_k=0
        )
        
        # Create runner
        runner = model_runner.ModelRunner(model_config, inference_config)
        
        # Mock tokenizer
        class MockTokenizer:
            def encode(self, text):
                return [1, 2, 3]
            
            def decode(self, tokens):
                return f"Result: {' '.join(str(t) for t in tokens)}"
            
            @property
            def eos_token_id(self):
                return 0
        
        runner.set_tokenizer(MockTokenizer())
        
        # Create and process request
        request = model_runner.GenerationRequest(
            request_id="test_basic",
            prompt="Test",
            max_new_tokens=3,
            temperature=0.5
        )
        
        runner.add_request(request)
        result = runner.process_request("test_basic")
        
        print(f"✓ Basic generation test passed")
        print(f"  Generated tokens: {result.generated_tokens}")
        print(f"  Generated text: {result.generated_text}")
        print(f"  Metrics: {result.metrics}")
        return True
    except Exception as e:
        print(f"✗ Basic generation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    print("=" * 60)
    print("Mini-vLLM Simple Test")
    print("=" * 60)
    
    tests = [
        test_engine_creation,
        test_model_runner,
        test_basic_generation
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print("\n" + "=" * 60)
    print(f"Test Results: {passed}/{total} passed")
    
    if passed == total:
        print("🎉 All tests passed!")
        return 0
    else:
        print("❌ Some tests failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())