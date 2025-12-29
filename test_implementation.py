#!/usr/bin/env python3

"""
Test script to verify the Mini-vLLM implementation
"""

import sys
import os

# Add the zllm directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'zllm'))

def test_imports():
    """Test that all modules can be imported"""
    print("Testing imports...")
    
    try:
        # Test Python modules
        from zllm.python.engine import InferenceEngine, ModelConfig, InferenceConfig
        from zllm.python.model_runner import ModelRunner, GenerationRequest
        print("✓ Python modules imported successfully")
        return True
    except ImportError as e:
        print(f"✗ Failed to import Python modules: {e}")
        return False

def test_cuda_bindings():
    """Test CUDA bindings if available"""
    print("\nTesting CUDA bindings...")
    
    try:
        import mini_vllm_cuda
        print("✓ CUDA module imported successfully")
        print(f"  CUDA version: {mini_vllm_cuda.cuda_version}")
        print(f"  Module version: {mini_vllm_cuda.__version__}")
        return True
    except ImportError:
        print("✗ CUDA module not available (expected if not built)")
        return False

def test_engine_creation():
    """Test engine creation"""
    print("\nTesting engine creation...")
    
    try:
        from zllm.python.engine import InferenceEngine, ModelConfig, InferenceConfig
        
        # Create small config for testing
        model_config = ModelConfig(
            num_layers=2,
            num_heads=4,
            num_kv_heads=1,
            head_dim=32,
            hidden_dim=128,
            intermediate_dim=256,
            vocab_size=100,
            max_seq_len=64,
            dtype="fp32"
        )
        
        inference_config = InferenceConfig(
            max_tokens=50,
            temperature=0.8,
            top_p=0.95,
            top_k=50
        )
        
        # Create engine
        engine = InferenceEngine(model_config, inference_config)
        print("✓ Engine created successfully")
        return True
    except Exception as e:
        print(f"✗ Failed to create engine: {e}")
        return False

def test_model_runner():
    """Test model runner"""
    print("\nTesting model runner...")
    
    try:
        from zllm.python.model_runner import ModelRunner, GenerationRequest
        from zllm.python.engine import ModelConfig, InferenceConfig
        
        # Create configs
        model_config = ModelConfig(
            num_layers=2,
            num_heads=4,
            num_kv_heads=1,
            head_dim=32,
            hidden_dim=128,
            intermediate_dim=256,
            vocab_size=100,
            max_seq_len=64,
            dtype="fp32"
        )
        
        inference_config = InferenceConfig(
            max_tokens=50,
            temperature=0.8,
            top_p=0.95,
            top_k=50
        )
        
        # Create runner
        runner = ModelRunner(model_config, inference_config)
        
        # Mock tokenizer
        class MockTokenizer:
            def encode(self, text):
                return [1, 2, 3, 4, 5]
            
            def decode(self, tokens):
                return f"Generated: {' '.join(str(t) for t in tokens)}"
            
            @property
            def eos_token_id(self):
                return 0
        
        runner.set_tokenizer(MockTokenizer())
        
        # Create and add request
        request = GenerationRequest(
            request_id="test_001",
            prompt="Hello, world!",
            max_new_tokens=5,
            temperature=0.8
        )
        
        runner.add_request(request)
        print("✓ Model runner created and request added successfully")
        return True
    except Exception as e:
        print(f"✗ Failed to test model runner: {e}")
        return False

def test_basic_functionality():
    """Test basic functionality without CUDA"""
    print("\nTesting basic functionality...")
    
    try:
        from zllm.python.model_runner import ModelRunner, GenerationRequest
        from zllm.python.engine import ModelConfig, InferenceConfig
        
        # Create configs
        model_config = ModelConfig(
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
        
        inference_config = InferenceConfig(
            max_tokens=10,
            temperature=0.5,
            top_p=1.0,
            top_k=0
        )
        
        # Create runner
        runner = ModelRunner(model_config, inference_config)
        
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
        request = GenerationRequest(
            request_id="test_basic",
            prompt="Test",
            max_new_tokens=3,
            temperature=0.5
        )
        
        runner.add_request(request)
        result = runner.process_request("test_basic")
        
        print(f"✓ Basic functionality test passed")
        print(f"  Generated tokens: {result.generated_tokens}")
        print(f"  Generated text: {result.generated_text}")
        return True
    except Exception as e:
        print(f"✗ Basic functionality test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    print("=" * 60)
    print("Mini-vLLM Implementation Test")
    print("=" * 60)
    
    tests = [
        test_imports,
        test_cuda_bindings,
        test_engine_creation,
        test_model_runner,
        test_basic_functionality
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