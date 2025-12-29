# =============================================================================
# model_runner.py - Model Runner for Mini-vLLM
# =============================================================================

import numpy as np
import torch
from typing import List, Dict, Optional, Tuple, Callable
from dataclasses import dataclass
import time
import asyncio
from collections import defaultdict

# Local imports
try:
    from .engine import InferenceEngine, ModelConfig, InferenceConfig
except ImportError:
    from engine import InferenceEngine, ModelConfig, InferenceConfig


@dataclass
class GenerationRequest:
    """Request for text generation"""
    request_id: str
    prompt: str
    max_new_tokens: int = 50
    temperature: float = 0.8
    top_p: float = 0.95
    top_k: int = 50
    callback: Optional[Callable] = None


@dataclass
class GenerationResult:
    """Result of text generation"""
    request_id: str
    generated_text: str
    generated_tokens: List[int]
    finish_reason: str  # "stop", "length", "error"
    metrics: Dict[str, float]


class ModelRunner:
    """Model runner that manages multiple sequences and handles generation requests"""
    
    def __init__(self, model_config: ModelConfig, inference_config: InferenceConfig):
        self.model_config = model_config
        self.inference_config = inference_config
        self.engine = InferenceEngine(model_config, inference_config)
        
        # Request management
        self.active_requests = {}  # request_id -> GenerationRequest
        self.completed_requests = {}  # request_id -> GenerationResult
        self.sequence_to_request = {}  # seq_id -> request_id
        
        # Statistics
        self.total_tokens_generated = 0
        self.total_requests = 0
        self.start_time = time.time()
        
        # Tokenizer (would be initialized with actual tokenizer)
        self.tokenizer = None
    
    def set_tokenizer(self, tokenizer):
        """Set the tokenizer to use"""
        self.tokenizer = tokenizer
    
    def add_request(self, request: GenerationRequest) -> str:
        """Add a new generation request"""
        if request.request_id in self.active_requests:
            raise ValueError(f"Request ID {request.request_id} already exists")
        
        # Store request
        self.active_requests[request.request_id] = request
        self.total_requests += 1
        
        # Add sequence to engine
        seq_id = self.engine.add_sequence(request.prompt, self.tokenizer)
        self.sequence_to_request[seq_id] = request.request_id
        
        return request.request_id
    
    def process_request(self, request_id: str) -> GenerationResult:
        """Process a single request synchronously"""
        if request_id not in self.active_requests:
            raise ValueError(f"Request ID {request_id} not found")
        
        request = self.active_requests[request_id]
        seq_id = None
        
        # Find the sequence ID for this request
        for sid, rid in self.sequence_to_request.items():
            if rid == request_id:
                seq_id = sid
                break
        
        if seq_id is None:
            raise ValueError(f"No sequence found for request {request_id}")
        
        # Update inference config for this request
        old_config = self.engine.inference_config
        self.engine.inference_config.temperature = request.temperature
        self.engine.inference_config.top_p = request.top_p
        self.engine.inference_config.top_k = request.top_k
        
        # Generate tokens
        start_time = time.time()
        generated_tokens = self.engine.generate(
            seq_id, 
            request.max_new_tokens, 
            self.tokenizer,
            callback=request.callback
        )
        generation_time = time.time() - start_time
        
        # Restore original config
        self.engine.inference_config = old_config
        
        # Create result
        result = GenerationResult(
            request_id=request_id,
            generated_text=self._tokens_to_text(generated_tokens),
            generated_tokens=generated_tokens,
            finish_reason="length" if len(generated_tokens) == request.max_new_tokens else "stop",
            metrics={
                "generation_time": generation_time,
                "tokens_per_second": len(generated_tokens) / generation_time,
                "total_tokens": len(generated_tokens)
            }
        )
        
        # Clean up
        self._complete_request(request_id, result)
        
        return result
    
    def _tokens_to_text(self, tokens: List[int]) -> str:
        """Convert tokens to text using the tokenizer"""
        if self.tokenizer is None:
            return f"[Tokens: {tokens}]"
        return self.tokenizer.decode(tokens)
    
    def _complete_request(self, request_id: str, result: GenerationResult):
        """Mark a request as completed"""
        # Remove from active requests
        if request_id in self.active_requests:
            del self.active_requests[request_id]
        
        # Store in completed requests
        self.completed_requests[request_id] = result
        
        # Update statistics
        self.total_tokens_generated += len(result.generated_tokens)
    
    def get_request_status(self, request_id: str) -> Dict:
        """Get status of a request"""
        if request_id in self.active_requests:
            return {"status": "active", "request": self.active_requests[request_id]}
        elif request_id in self.completed_requests:
            return {"status": "completed", "result": self.completed_requests[request_id]}
        else:
            return {"status": "not_found"}
    
    def get_stats(self) -> Dict:
        """Get runner statistics"""
        uptime = time.time() - self.start_time
        return {
            "active_requests": len(self.active_requests),
            "completed_requests": len(self.completed_requests),
            "total_requests": self.total_requests,
            "total_tokens_generated": self.total_tokens_generated,
            "uptime_seconds": uptime,
            "tokens_per_second": self.total_tokens_generated / uptime if uptime > 0 else 0
        }


class AsyncModelRunner:
    """Asynchronous version of ModelRunner for concurrent processing"""
    
    def __init__(self, model_config: ModelConfig, inference_config: InferenceConfig):
        self.model_config = model_config
        self.inference_config = inference_config
        self.engine = InferenceEngine(model_config, inference_config)
        
        # Request management
        self.active_requests = {}  # request_id -> GenerationRequest
        self.completed_requests = {}  # request_id -> GenerationResult
        self.sequence_to_request = {}  # seq_id -> request_id
        self.request_queue = asyncio.Queue()
        
        # Statistics
        self.total_tokens_generated = 0
        self.total_requests = 0
        self.start_time = time.time()
        
        # Tokenizer
        self.tokenizer = None
        
        # Processing flag
        self._processing = False
    
    def set_tokenizer(self, tokenizer):
        """Set the tokenizer to use"""
        self.tokenizer = tokenizer
    
    def add_request(self, request: GenerationRequest) -> str:
        """Add a new generation request to the queue"""
        if request.request_id in self.active_requests:
            raise ValueError(f"Request ID {request.request_id} already exists")
        
        # Store request
        self.active_requests[request.request_id] = request
        self.total_requests += 1
        
        # Add to queue
        self.request_queue.put_nowait(request)
        
        # Start processing if not already running
        if not self._processing:
            self._processing = True
            asyncio.create_task(self._process_queue())
        
        return request.request_id
    
    async def _process_queue(self):
        """Process requests from the queue"""
        while not self.request_queue.empty():
            request = await self.request_queue.get()
            
            try:
                # Add sequence to engine
                seq_id = self.engine.add_sequence(request.prompt, self.tokenizer)
                self.sequence_to_request[seq_id] = request.request_id
                
                # Process the request
                result = await self._process_request_async(request, seq_id)
                
                # Store result
                self.completed_requests[request.request_id] = result
                
            except Exception as e:
                result = GenerationResult(
                    request_id=request.request_id,
                    generated_text="",
                    generated_tokens=[],
                    finish_reason="error",
                    metrics={"error": str(e)}
                )
                self.completed_requests[request.request_id] = result
            
            finally:
                # Clean up
                if request.request_id in self.active_requests:
                    del self.active_requests[request.request_id]
        
        self._processing = False
    
    async def _process_request_async(self, request: GenerationRequest, seq_id: int) -> GenerationResult:
        """Process a single request asynchronously"""
        # Update inference config for this request
        old_config = self.engine.inference_config
        self.engine.inference_config.temperature = request.temperature
        self.engine.inference_config.top_p = request.top_p
        self.engine.inference_config.top_k = request.top_k
        
        # Generate tokens
        start_time = time.time()
        generated_tokens = []
        
        # Prefill phase
        if len(self.engine.sequences[seq_id]['generated_tokens']) == 0:
            logits = self.engine.prefill(seq_id)
        else:
            # Continue from last token
            last_token = self.engine.sequences[seq_id]['input_ids'][-1]
            logits = self.engine.decode(seq_id, last_token)
        
        # Generate new tokens
        for i in range(request.max_new_tokens):
            # Sample next token
            next_token = self._sample(logits)
            
            if request.callback:
                await request.callback(next_token)
            
            generated_tokens.append(next_token)
            
            # Check for stop conditions
            if next_token == self.tokenizer.eos_token_id:
                break
            
            # Decode next token
            logits = self.engine.decode(seq_id, next_token)
            
            # Yield control to event loop occasionally
            if i % 10 == 0:
                await asyncio.sleep(0)
        
        generation_time = time.time() - start_time
        
        # Restore original config
        self.engine.inference_config = old_config
        
        # Create result
        result = GenerationResult(
            request_id=request.request_id,
            generated_text=self._tokens_to_text(generated_tokens),
            generated_tokens=generated_tokens,
            finish_reason="length" if len(generated_tokens) == request.max_new_tokens else "stop",
            metrics={
                "generation_time": generation_time,
                "tokens_per_second": len(generated_tokens) / generation_time,
                "total_tokens": len(generated_tokens)
            }
        )
        
        # Update statistics
        self.total_tokens_generated += len(generated_tokens)
        
        return result
    
    def _sample(self, logits: np.ndarray) -> int:
        """Sample next token from logits"""
        config = self.engine.inference_config
        
        # Apply temperature
        if config.temperature != 1.0:
            logits = logits / config.temperature
        
        # Apply top-k filtering
        if config.top_k > 0:
            top_k_indices = np.argpartition(logits, -config.top_k)[-config.top_k:]
            mask = np.zeros_like(logits, dtype=bool)
            mask[top_k_indices] = True
            logits[~mask] = -float('inf')
        
        # Apply top-p filtering
        if config.top_p < 1.0:
            sorted_logits = np.sort(logits)[::-1]
            cumulative_probs = np.cumsum(np.exp(sorted_logits - np.max(sorted_logits)))
            cumulative_probs = cumulative_probs / cumulative_probs[-1]
            
            # Remove tokens with cumulative probability above the threshold
            mask = cumulative_probs > config.top_p
            if np.any(mask):
                threshold = sorted_logits[np.where(mask)[0][0]]
                logits[logits < threshold] = -float('inf')
        
        # Convert to probabilities
        probs = np.exp(logits - np.max(logits))
        probs = probs / np.sum(probs)
        
        # Sample
        next_token = np.random.choice(len(probs), p=probs)
        
        return next_token
    
    def _tokens_to_text(self, tokens: List[int]) -> str:
        """Convert tokens to text using the tokenizer"""
        if self.tokenizer is None:
            return f"[Tokens: {tokens}]"
        return self.tokenizer.decode(tokens)
    
    def get_request_status(self, request_id: str) -> Dict:
        """Get status of a request"""
        if request_id in self.active_requests:
            return {"status": "active", "request": self.active_requests[request_id]}
        elif request_id in self.completed_requests:
            return {"status": "completed", "result": self.completed_requests[request_id]}
        else:
            return {"status": "not_found"}
    
    def get_stats(self) -> Dict:
        """Get runner statistics"""
        uptime = time.time() - self.start_time
        return {
            "active_requests": len(self.active_requests),
            "completed_requests": len(self.completed_requests),
            "total_requests": self.total_requests,
            "total_tokens_generated": self.total_tokens_generated,
            "uptime_seconds": uptime,
            "tokens_per_second": self.total_tokens_generated / uptime if uptime > 0 else 0
        }


class BatchModelRunner:
    """Batch model runner that processes multiple requests in parallel"""
    
    def __init__(self, model_config: ModelConfig, inference_config: InferenceConfig, 
                 max_batch_size: int = 8):
        self.model_config = model_config
        self.inference_config = inference_config
        self.engine = InferenceEngine(model_config, inference_config)
        self.max_batch_size = max_batch_size
        
        # Request management
        self.active_requests = {}  # request_id -> GenerationRequest
        self.completed_requests = {}  # request_id -> GenerationResult
        self.sequence_to_request = {}  # seq_id -> request_id
        
        # Statistics
        self.total_tokens_generated = 0
        self.total_requests = 0
        self.start_time = time.time()
        
        # Tokenizer
        self.tokenizer = None
    
    def set_tokenizer(self, tokenizer):
        """Set the tokenizer to use"""
        self.tokenizer = tokenizer
    
    def add_request(self, request: GenerationRequest) -> str:
        """Add a new generation request"""
        if request.request_id in self.active_requests:
            raise ValueError(f"Request ID {request.request_id} already exists")
        
        # Store request
        self.active_requests[request.request_id] = request
        self.total_requests += 1
        
        # Add sequence to engine
        seq_id = self.engine.add_sequence(request.prompt, self.tokenizer)
        self.sequence_to_request[seq_id] = request.request_id
        
        return request.request_id
    
    def process_batch(self) -> List[GenerationResult]:
        """Process a batch of requests"""
        if not self.active_requests:
            return []
        
        # Get up to max_batch_size requests
        batch_requests = list(self.active_requests.items())[:self.max_batch_size]
        results = []
        
        for request_id, request in batch_requests:
            try:
                # Find the sequence ID for this request
                seq_id = None
                for sid, rid in self.sequence_to_request.items():
                    if rid == request_id:
                        seq_id = sid
                        break
                
                if seq_id is None:
                    raise ValueError(f"No sequence found for request {request_id}")
                
                # Process the request
                result = self._process_single_request(request, seq_id)
                results.append(result)
                
            except Exception as e:
                result = GenerationResult(
                    request_id=request_id,
                    generated_text="",
                    generated_tokens=[],
                    finish_reason="error",
                    metrics={"error": str(e)}
                )
                results.append(result)
        
        return results
    
    def _process_single_request(self, request: GenerationRequest, seq_id: int) -> GenerationResult:
        """Process a single request"""
        # Update inference config for this request
        old_config = self.engine.inference_config
        self.engine.inference_config.temperature = request.temperature
        self.engine.inference_config.top_p = request.top_p
        self.engine.inference_config.top_k = request.top_k
        
        # Generate tokens
        start_time = time.time()
        generated_tokens = self.engine.generate(
            seq_id, 
            request.max_new_tokens, 
            self.tokenizer,
            callback=request.callback
        )
        generation_time = time.time() - start_time
        
        # Restore original config
        self.engine.inference_config = old_config
        
        # Create result
        result = GenerationResult(
            request_id=request.request_id,
            generated_text=self._tokens_to_text(generated_tokens),
            generated_tokens=generated_tokens,
            finish_reason="length" if len(generated_tokens) == request.max_new_tokens else "stop",
            metrics={
                "generation_time": generation_time,
                "tokens_per_second": len(generated_tokens) / generation_time,
                "total_tokens": len(generated_tokens)
            }
        )
        
        # Clean up
        self._complete_request(request.request_id, result)
        
        return result
    
    def _complete_request(self, request_id: str, result: GenerationResult):
        """Mark a request as completed"""
        # Remove from active requests
        if request_id in self.active_requests:
            del self.active_requests[request_id]
        
        # Store in completed requests
        self.completed_requests[request_id] = result
        
        # Update statistics
        self.total_tokens_generated += len(result.generated_tokens)
    
    def _tokens_to_text(self, tokens: List[int]) -> str:
        """Convert tokens to text using the tokenizer"""
        if self.tokenizer is None:
            return f"[Tokens: {tokens}]"
        return self.tokenizer.decode(tokens)
    
    def get_stats(self) -> Dict:
        """Get runner statistics"""
        uptime = time.time() - self.start_time
        return {
            "active_requests": len(self.active_requests),
            "completed_requests": len(self.completed_requests),
            "total_requests": self.total_requests,
            "total_tokens_generated": self.total_tokens_generated,
            "uptime_seconds": uptime,
            "tokens_per_second": self.total_tokens_generated / uptime if uptime > 0 else 0,
            "max_batch_size": self.max_batch_size
        }


def test_model_runner():
    """Test the model runner"""
    print("Testing Model Runner...")
    
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
        max_tokens=100,
        temperature=0.7,
        top_p=0.9,
        top_k=30
    )
    
    # Create model runner
    runner = ModelRunner(model_config, inference_config)
    
    # Mock tokenizer
    class MockTokenizer:
        def encode(self, text):
            return [1, 2, 3, 4, 5]  # Simple mock
        
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
        max_new_tokens=10,
        temperature=0.8
    )
    
    runner.add_request(request)
    
    # Process request
    result = runner.process_request("test_001")
    
    print(f"Request completed: {result.request_id}")
    print(f"Generated text: {result.generated_text}")
    print(f"Generated tokens: {result.generated_tokens}")
    print(f"Finish reason: {result.finish_reason}")
    print(f"Metrics: {result.metrics}")
    
    # Get stats
    stats = runner.get_stats()
    print(f"Runner stats: {stats}")
    
    print("Test completed!")


if __name__ == "__main__":
    test_model_runner()