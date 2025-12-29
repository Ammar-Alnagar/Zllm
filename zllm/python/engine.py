# =============================================================================
# engine.py - Mini-vLLM Inference Engine
# =============================================================================

import numpy as np
import torch
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import time

# Try to import our CUDA module
try:
    import mini_vllm_cuda
    CUDA_AVAILABLE = True
except ImportError:
    CUDA_AVAILABLE = False
    print("Warning: CUDA module not available, falling back to CPU")


@dataclass
class ModelConfig:
    """Configuration for the transformer model"""
    num_layers: int = 32
    num_heads: int = 32
    num_kv_heads: int = 8
    head_dim: int = 128
    hidden_dim: int = 4096  # num_heads * head_dim
    intermediate_dim: int = 11008  # (hidden_dim * 8) // 3
    vocab_size: int = 151936
    max_seq_len: int = 4096
    block_size: int = 16  # Tokens per KV cache block
    dtype: str = "fp16"  # or "fp32"


@dataclass
class InferenceConfig:
    """Configuration for inference"""
    max_tokens: int = 512
    temperature: float = 0.8
    top_p: float = 0.95
    top_k: int = 50
    repetition_penalty: float = 1.1
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0


class KVCache:
    """Key-Value cache for attention"""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.num_layers = config.num_layers
        self.num_kv_heads = config.num_kv_heads
        self.head_dim = config.head_dim
        self.block_size = config.block_size
        self.dtype = config.dtype
        
        # Initialize cache
        self.key_cache = []
        self.value_cache = []
        self.block_table = []  # Maps sequence -> list of block IDs
        self.current_lengths = []  # Current length of each sequence
        
        # Memory management
        self.max_blocks = 1000  # Start with reasonable size
        
        if CUDA_AVAILABLE:
            # Initialize GPU memory
            self._init_gpu_memory()
    
    def _init_gpu_memory(self):
        """Initialize GPU memory for KV cache"""
        # Calculate total memory needed
        block_size_bytes = 2 * self.num_kv_heads * self.block_size * self.head_dim
        if self.dtype == "fp16":
            block_size_bytes *= 2  # 2 bytes per FP16
        else:
            block_size_bytes *= 4  # 4 bytes per FP32
            
        total_memory_gb = self.max_blocks * block_size_bytes / (1024 ** 3)
        
        # Initialize memory manager
        self.memory_manager = mini_vllm_cuda.MemoryManager()
        self.memory_manager.init(
            kv_cache_gb=total_memory_gb,
            block_size_tokens=self.block_size,
            num_kv_heads=self.num_kv_heads,
            head_dim=self.head_dim,
            dtype_size=2 if self.dtype == "fp16" else 4
        )
    
    def allocate_sequence(self, seq_id: int, initial_length: int = 0):
        """Allocate a new sequence in the cache"""
        if seq_id >= len(self.current_lengths):
            # Extend arrays if needed
            while len(self.current_lengths) <= seq_id:
                self.current_lengths.append(0)
                self.block_table.append([])
        
        self.current_lengths[seq_id] = initial_length
        
        # Allocate initial blocks
        num_blocks_needed = (initial_length + self.block_size - 1) // self.block_size
        for _ in range(num_blocks_needed):
            block_id = self._allocate_block()
            self.block_table[seq_id].append(block_id)
    
    def _allocate_block(self) -> int:
        """Allocate a new block from the memory pool"""
        if CUDA_AVAILABLE:
            # Use GPU memory manager
            block_id = len(self.key_cache)
            self.key_cache.append(None)  # Placeholder
            self.value_cache.append(None)  # Placeholder
            return block_id
        else:
            # CPU fallback
            block_id = len(self.key_cache)
            
            # Allocate numpy arrays for the block
            if self.dtype == "fp16":
                dtype = np.float16
            else:
                dtype = np.float32
            
            self.key_cache.append(
                np.zeros((self.num_kv_heads, self.block_size, self.head_dim), dtype=dtype)
            )
            self.value_cache.append(
                np.zeros((self.num_kv_heads, self.block_size, self.head_dim), dtype=dtype)
            )
            
            return block_id
    
    def get_cache_for_inference(self, seq_id: int, layer: int) -> Tuple[np.ndarray, np.ndarray]:
        """Get the cache tensors for a specific sequence and layer"""
        if CUDA_AVAILABLE:
            # Return GPU pointers (simplified for now)
            # In a real implementation, we'd return the actual GPU memory pointers
            return None, None
        else:
            # CPU fallback - concatenate all blocks for this sequence
            blocks = self.block_table[seq_id]
            current_length = self.current_lengths[seq_id]
            
            # Calculate how many full blocks we need
            num_full_blocks = current_length // self.block_size
            remaining_tokens = current_length % self.block_size
            
            # Concatenate key cache
            key_blocks = [self.key_cache[b][:, :self.block_size, :] for b in blocks[:num_full_blocks]]
            if remaining_tokens > 0 and num_full_blocks < len(blocks):
                key_blocks.append(self.key_cache[blocks[num_full_blocks]][:, :remaining_tokens, :])
            
            # Concatenate value cache
            value_blocks = [self.value_cache[b][:, :self.block_size, :] for b in blocks[:num_full_blocks]]
            if remaining_tokens > 0 and num_full_blocks < len(blocks):
                value_blocks.append(self.value_cache[blocks[num_full_blocks]][:, :remaining_tokens, :])
            
            if key_blocks:
                key_cache = np.concatenate(key_blocks, axis=1)
                value_cache = np.concatenate(value_blocks, axis=1)
            else:
                key_cache = np.zeros((self.num_kv_heads, 0, self.head_dim), dtype=self.key_cache[0].dtype)
                value_cache = np.zeros((self.num_kv_heads, 0, self.head_dim), dtype=self.value_cache[0].dtype)
            
            return key_cache, value_cache
    
    def update_cache(self, seq_id: int, layer: int, 
                    new_keys: np.ndarray, new_values: np.ndarray):
        """Update the cache with new key/value pairs"""
        current_length = self.current_lengths[seq_id]
        block_idx = current_length // self.block_size
        pos_in_block = current_length % self.block_size
        
        # Allocate new block if needed
        if pos_in_block == 0:
            new_block_id = self._allocate_block()
            self.block_table[seq_id].append(new_block_id)
        
        # Update the cache
        block_id = self.block_table[seq_id][block_idx]
        
        if CUDA_AVAILABLE:
            # GPU update (simplified)
            pass
        else:
            # CPU update
            self.key_cache[block_id][:, pos_in_block:pos_in_block+1, :] = new_keys
            self.value_cache[block_id][:, pos_in_block:pos_in_block+1, :] = new_values
        
        # Update current length
        self.current_lengths[seq_id] += 1


class TransformerLayer:
    """Single transformer layer"""
    
    def __init__(self, config: ModelConfig, layer_id: int):
        self.config = config
        self.layer_id = layer_id
        
        # Initialize weights (random for now, would be loaded from model)
        self._init_weights()
        
        # RoPE tables
        self._init_rope_tables()
    
    def _init_weights(self):
        """Initialize layer weights"""
        # Attention weights
        self.attn_wq = np.random.randn(self.config.hidden_dim, self.config.num_heads * self.config.head_dim).astype(np.float32)
        self.attn_wk = np.random.randn(self.config.hidden_dim, self.config.num_kv_heads * self.config.head_dim).astype(np.float32)
        self.attn_wv = np.random.randn(self.config.hidden_dim, self.config.num_kv_heads * self.config.head_dim).astype(np.float32)
        self.attn_wo = np.random.randn(self.config.num_heads * self.config.head_dim, self.config.hidden_dim).astype(np.float32)
        
        # FFN weights
        self.ffn_w1 = np.random.randn(self.config.hidden_dim, self.config.intermediate_dim).astype(np.float32)
        self.ffn_w2 = np.random.randn(self.config.intermediate_dim, self.config.hidden_dim).astype(np.float32)
        self.ffn_w3 = np.random.randn(self.config.hidden_dim, self.config.intermediate_dim).astype(np.float32)
        
        # Layer norms
        self.attn_norm_weight = np.ones(self.config.hidden_dim, dtype=np.float32)
        self.ffn_norm_weight = np.ones(self.config.hidden_dim, dtype=np.float32)
    
    def _init_rope_tables(self):
        """Initialize RoPE tables"""
        if CUDA_AVAILABLE:
            cos_table, sin_table = mini_vllm_cuda.rope_init_tables(
                self.config.max_seq_len, 
                self.config.head_dim
            )
            self.cos_table = np.array(cos_table)
            self.sin_table = np.array(sin_table)
        else:
            # CPU implementation
            self.cos_table = np.zeros((self.config.max_seq_len, self.config.head_dim // 2))
            self.sin_table = np.zeros((self.config.max_seq_len, self.config.head_dim // 2))
            
            for pos in range(self.config.max_seq_len):
                for dim in range(self.config.head_dim // 2):
                    freq = 10000.0 ** (-2.0 * dim / self.config.head_dim)
                    angle = pos * freq
                    self.cos_table[pos, dim] = np.cos(angle)
                    self.sin_table[pos, dim] = np.sin(angle)
    
    def forward(self, hidden_states: np.ndarray, kv_cache: KVCache, 
                seq_id: int, positions: np.ndarray, is_prefill: bool) -> np.ndarray:
        """Forward pass through the transformer layer"""
        
        # Layer norm before attention
        attn_input = self._rms_norm(hidden_states, self.attn_norm_weight)
        
        # Self-attention
        attn_output = self._self_attention(attn_input, kv_cache, seq_id, positions, is_prefill)
        
        # Residual connection
        hidden_states = hidden_states + attn_output
        
        # Layer norm before FFN
        ffn_input = self._rms_norm(hidden_states, self.ffn_norm_weight)
        
        # Feed-forward network
        ffn_output = self._feed_forward(ffn_input)
        
        # Residual connection
        hidden_states = hidden_states + ffn_output
        
        return hidden_states
    
    def _rms_norm(self, x: np.ndarray, weight: np.ndarray) -> np.ndarray:
        """RMS normalization"""
        if CUDA_AVAILABLE:
            # Use GPU implementation
            x_tensor = torch.from_numpy(x).float()
            weight_tensor = torch.from_numpy(weight).float()
            
            # Convert to numpy arrays that pybind11 can handle
            x_np = x_tensor.numpy()
            weight_np = weight_tensor.numpy()
            
            result = mini_vllm_cuda.rmsnorm_forward(x_np, weight_np)
            return np.array(result)
        else:
            # CPU implementation
            x_squared = x ** 2
            mean_squared = np.mean(x_squared, axis=-1, keepdims=True)
            rms = np.sqrt(mean_squared + 1e-6)
            return (x / rms) * weight
    
    def _self_attention(self, x: np.ndarray, kv_cache: KVCache, seq_id: int, 
                       positions: np.ndarray, is_prefill: bool) -> np.ndarray:
        """Self-attention mechanism"""
        
        # Project to Q, K, V
        q = self._linear(x, self.attn_wq)  # [batch, num_heads, head_dim]
        k = self._linear(x, self.attn_wk)  # [batch, num_kv_heads, head_dim]
        v = self._linear(x, self.attn_wv)  # [batch, num_kv_heads, head_dim]
        
        # Reshape for multi-head attention
        batch_size = x.shape[0]
        q = q.reshape(batch_size, self.config.num_heads, self.config.head_dim)
        k = k.reshape(batch_size, self.config.num_kv_heads, self.config.head_dim)
        v = v.reshape(batch_size, self.config.num_kv_heads, self.config.head_dim)
        
        # Apply RoPE
        if CUDA_AVAILABLE:
            # Use GPU RoPE
            q_flat = q.reshape(-1, self.config.head_dim)
            k_flat = k.reshape(-1, self.config.head_dim)
            
            q_rope, k_rope = mini_vllm_cuda.rope_forward(
                q_flat, k_flat, 
                self.cos_table, self.sin_table,
                positions, 
                self.config.num_heads, self.config.num_kv_heads,
                self.config.head_dim
            )
            
            q = q_rope.reshape(batch_size, self.config.num_heads, self.config.head_dim)
            k = k_rope.reshape(batch_size, self.config.num_kv_heads, self.config.head_dim)
        else:
            # CPU RoPE
            q, k = self._apply_rope_cpu(q, k, positions)
        
        # Attention computation
        if is_prefill:
            attn_output = self._flash_attention_prefill(q, k, v, positions)
        else:
            attn_output = self._flash_attention_decode(q, k, v, kv_cache, seq_id, positions)
        
        # Update KV cache (only for the last token in decode phase)
        if not is_prefill and batch_size == 1:
            # For decode phase, update cache with the new K and V
            kv_cache.update_cache(seq_id, self.layer_id, k, v)
        
        # Project back
        attn_output = attn_output.reshape(batch_size, -1)  # Flatten heads
        output = self._linear(attn_output, self.attn_wo)
        
        return output
    
    def _apply_rope_cpu(self, q: np.ndarray, k: np.ndarray, positions: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Apply RoPE on CPU"""
        for i in range(q.shape[0]):
            pos = positions[i]
            head_dim = q.shape[-1]
            half_head_dim = head_dim // 2
            
            for dim in range(half_head_dim):
                # Get the rotation angle for this position and dimension
                cos_val = self.cos_table[pos, dim]
                sin_val = self.sin_table[pos, dim]
                
                # Apply rotation to both Q and K
                for head_idx in range(q.shape[1]):
                    # Q rotation
                    q_even = q[i, head_idx, 2*dim]
                    q_odd = q[i, head_idx, 2*dim+1]
                    q[i, head_idx, 2*dim] = q_even * cos_val - q_odd * sin_val
                    q[i, head_idx, 2*dim+1] = q_even * sin_val + q_odd * cos_val
                    
                for head_idx in range(k.shape[1]):
                    # K rotation
                    k_even = k[i, head_idx, 2*dim]
                    k_odd = k[i, head_idx, 2*dim+1]
                    k[i, head_idx, 2*dim] = k_even * cos_val - k_odd * sin_val
                    k[i, head_idx, 2*dim+1] = k_even * sin_val + k_odd * cos_val
        
        return q, k
    
    def _flash_attention_prefill(self, q: np.ndarray, k: np.ndarray, v: np.ndarray, 
                                positions: np.ndarray) -> np.ndarray:
        """Flash attention for prefill phase"""
        if CUDA_AVAILABLE:
            # Use GPU flash attention
            config = mini_vllm_cuda.FlashAttentionConfig()
            config.num_heads = self.config.num_heads
            config.num_kv_heads = self.config.num_kv_heads
            config.head_dim = self.config.head_dim
            config.max_seq_len = self.config.max_seq_len
            config.softmax_scale = 1.0 / np.sqrt(self.config.head_dim)
            config.is_causal = True
            config.dtype_size = 2 if self.config.dtype == "fp16" else 4
            
            # Convert to numpy arrays
            q_np = q.astype(np.float32)
            k_np = k.astype(np.float32)
            v_np = v.astype(np.float32)
            positions_np = positions.astype(np.int32)
            
            result = mini_vllm_cuda.flash_attention_prefill_forward(
                q_np, k_np, v_np, positions_np, config
            )
            return np.array(result)
        else:
            # CPU fallback - naive attention
            return self._naive_attention(q, k, v, positions)
    
    def _flash_attention_decode(self, q: np.ndarray, k: np.ndarray, v: np.ndarray,
                               kv_cache: KVCache, seq_id: int, positions: np.ndarray) -> np.ndarray:
        """Flash attention for decode phase"""
        if CUDA_AVAILABLE:
            # Use GPU flash infer
            config = mini_vllm_cuda.FlashInferConfig()
            config.num_heads = self.config.num_heads
            config.num_kv_heads = self.config.num_kv_heads
            config.head_dim = self.config.head_dim
            config.softmax_scale = 1.0 / np.sqrt(self.config.head_dim)
            config.dtype_size = 2 if self.config.dtype == "fp16" else 4
            
            # Get cache for this sequence
            key_cache, value_cache = kv_cache.get_cache_for_inference(seq_id, self.layer_id)
            
            # Prepare block table and offsets (simplified)
            current_length = kv_cache.current_lengths[seq_id]
            block_table = np.array([0], dtype=np.int32)  # Simplified
            block_offsets = np.array([current_length - 1], dtype=np.int32)
            seq_lengths = np.array([current_length], dtype=np.int32)
            
            # Convert to numpy arrays
            q_np = q.astype(np.float32)
            key_cache_np = key_cache.astype(np.float32)
            value_cache_np = value_cache.astype(np.float32)
            
            result = mini_vllm_cuda.flash_infer_forward(
                q_np, key_cache_np, value_cache_np,
                block_table, block_offsets, seq_lengths, config
            )
            return np.array(result)
        else:
            # CPU fallback - use cache
            key_cache, value_cache = kv_cache.get_cache_for_inference(seq_id, self.layer_id)
            
            # Handle empty cache case
            if key_cache.shape[1] == 0:
                # First token in sequence, just use the new K, V
                return self._naive_attention(q, k, v, positions)
            else:
                # Combine with new K, V
                k = np.concatenate([key_cache, k], axis=1)
                v = np.concatenate([value_cache, v], axis=1)
                
                return self._naive_attention(q, k, v, positions)
    
    def _naive_attention(self, q: np.ndarray, k: np.ndarray, v: np.ndarray, 
                        positions: np.ndarray) -> np.ndarray:
        """Naive attention implementation for CPU fallback"""
        batch_size = q.shape[0]
        num_heads = q.shape[1]
        head_dim = q.shape[2]
        seq_len = k.shape[1]
        
        # For GQA, we need to handle the case where num_heads != num_kv_heads
        # We'll repeat the K and V heads to match the number of Q heads
        num_kv_heads = k.shape[1]
        
        if num_heads != num_kv_heads:
            # Grouped Query Attention: repeat K and V heads
            repeat_factor = num_heads // num_kv_heads
            k = np.repeat(k, repeat_factor, axis=1)
            v = np.repeat(v, repeat_factor, axis=1)
        
        # Compute attention scores
        scores = np.einsum('bhi,bhj->bhij', q, k) / np.sqrt(head_dim)
        
        # Causal masking
        causal_mask = np.triu(np.ones((seq_len, seq_len)), k=1)
        scores = scores - causal_mask * 1e9
        
        # Softmax
        exp_scores = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
        attn_weights = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)
        
        # Apply attention
        output = np.einsum('bhij,bhj->bhi', attn_weights, v)
        
        return output
    
    def _feed_forward(self, x: np.ndarray) -> np.ndarray:
        """Feed-forward network with SwiGLU activation"""
        if CUDA_AVAILABLE:
            # Use GPU SwiGLU
            gate = self._linear(x, self.ffn_w1)
            up = self._linear(x, self.ffn_w3)
            
            # Convert to numpy arrays
            gate_np = gate.astype(np.float32)
            up_np = up.astype(np.float32)
            
            result = mini_vllm_cuda.swiglu_forward(gate_np, up_np)
            output = self._linear(np.array(result), self.ffn_w2)
        else:
            # CPU SwiGLU
            gate = self._linear(x, self.ffn_w1)
            up = self._linear(x, self.ffn_w3)
            
            # SiLU activation
            silu_gate = gate * (1.0 / (1.0 + np.exp(-gate)))
            output = silu_gate * up
            output = self._linear(output, self.ffn_w2)
        
        return output
    
    def _linear(self, x: np.ndarray, weight: np.ndarray) -> np.ndarray:
        """Linear transformation"""
        return np.dot(x, weight)


class TransformerModel:
    """Full transformer model"""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.layers = [TransformerLayer(config, i) for i in range(config.num_layers)]
        
        # Embedding and final layer
        self.embedding = np.random.randn(config.vocab_size, config.hidden_dim).astype(np.float32)
        self.final_norm_weight = np.ones(config.hidden_dim, dtype=np.float32)
        self.lm_head = np.random.randn(config.hidden_dim, config.vocab_size).astype(np.float32)
        
        # KV Cache
        self.kv_cache = KVCache(config)
    
    def forward(self, input_ids: np.ndarray, seq_id: int = 0, is_prefill: bool = True) -> np.ndarray:
        """Forward pass through the entire model"""
        
        # Embed tokens
        hidden_states = self.embedding[input_ids]
        
        # Create positions array
        positions = np.arange(len(input_ids))
        
        # Initialize cache for this sequence if needed
        if seq_id >= len(self.kv_cache.current_lengths):
            self.kv_cache.allocate_sequence(seq_id, len(input_ids))
        
        # Process through each layer
        for layer in self.layers:
            hidden_states = layer.forward(hidden_states, self.kv_cache, seq_id, positions, is_prefill)
        
        # Final layer norm
        hidden_states = self._rms_norm(hidden_states, self.final_norm_weight)
        
        # Language model head
        logits = np.dot(hidden_states, self.lm_head)
        
        return logits
    
    def _rms_norm(self, x: np.ndarray, weight: np.ndarray) -> np.ndarray:
        """RMS normalization"""
        if CUDA_AVAILABLE:
            x_tensor = torch.from_numpy(x).float()
            weight_tensor = torch.from_numpy(weight).float()
            
            x_np = x_tensor.numpy()
            weight_np = weight_tensor.numpy()
            
            result = mini_vllm_cuda.rmsnorm_forward(x_np, weight_np)
            return np.array(result)
        else:
            x_squared = x ** 2
            mean_squared = np.mean(x_squared, axis=-1, keepdims=True)
            rms = np.sqrt(mean_squared + 1e-6)
            return (x / rms) * weight


class InferenceEngine:
    """Main inference engine"""
    
    def __init__(self, model_config: ModelConfig, inference_config: InferenceConfig):
        self.model_config = model_config
        self.inference_config = inference_config
        self.model = TransformerModel(model_config)
        
        # Initialize sequence management
        self.sequences = {}  # seq_id -> sequence info
        self.next_seq_id = 0
    
    def add_sequence(self, prompt: str, tokenizer) -> int:
        """Add a new sequence to the engine"""
        seq_id = self.next_seq_id
        self.next_seq_id += 1
        
        # Tokenize prompt
        input_ids = tokenizer.encode(prompt)
        
        # Store sequence info
        self.sequences[seq_id] = {
            'input_ids': input_ids,
            'generated_tokens': [],
            'finished': False
        }
        
        # Allocate in KV cache
        self.model.kv_cache.allocate_sequence(seq_id, len(input_ids))
        
        return seq_id
    
    def prefill(self, seq_id: int) -> np.ndarray:
        """Prefill phase - process entire prompt"""
        if seq_id not in self.sequences:
            raise ValueError(f"Sequence {seq_id} not found")
        
        sequence = self.sequences[seq_id]
        input_ids = np.array(sequence['input_ids'])
        
        # Forward pass
        logits = self.model.forward(input_ids, seq_id, is_prefill=True)
        
        # Get last token logits
        last_logits = logits[-1]
        
        return last_logits
    
    def decode(self, seq_id: int, next_token: int) -> np.ndarray:
        """Decode phase - process one token"""
        if seq_id not in self.sequences:
            raise ValueError(f"Sequence {seq_id} not found")
        
        # Add token to sequence
        self.sequences[seq_id]['input_ids'].append(next_token)
        self.sequences[seq_id]['generated_tokens'].append(next_token)
        
        # Forward pass with single token
        input_ids = np.array([next_token])
        logits = self.model.forward(input_ids, seq_id, is_prefill=False)
        
        return logits[0]
    
    def generate(self, seq_id: int, max_new_tokens: int, tokenizer, 
                 callback=None) -> List[int]:
        """Generate tokens for a sequence"""
        if seq_id not in self.sequences:
            raise ValueError(f"Sequence {seq_id} not found")
        
        sequence = self.sequences[seq_id]
        generated_tokens = []
        
        # Prefill phase
        if len(sequence['generated_tokens']) == 0:
            logits = self.prefill(seq_id)
        else:
            # Continue from last token
            last_token = sequence['input_ids'][-1]
            logits = self.decode(seq_id, last_token)
        
        # Generate new tokens
        for i in range(max_new_tokens):
            # Sample next token
            next_token = self._sample(logits, tokenizer)
            
            if callback:
                callback(next_token)
            
            generated_tokens.append(next_token)
            
            # Check for stop conditions
            if next_token == tokenizer.eos_token_id:
                sequence['finished'] = True
                break
            
            # Decode next token
            logits = self.decode(seq_id, next_token)
        
        return generated_tokens
    
    def _sample(self, logits: np.ndarray, tokenizer) -> int:
        """Sample next token from logits"""
        # Apply temperature
        if self.inference_config.temperature != 1.0:
            logits = logits / self.inference_config.temperature
        
        # Apply top-k filtering
        if self.inference_config.top_k > 0:
            # Ensure top_k doesn't exceed vocabulary size
            effective_top_k = min(self.inference_config.top_k, len(logits))
            if effective_top_k > 0:
                top_k_indices = np.argpartition(logits, -effective_top_k)[-effective_top_k:]
                mask = np.zeros_like(logits, dtype=bool)
                mask[top_k_indices] = True
                logits[~mask] = -float('inf')
        
        # Apply top-p filtering
        if self.inference_config.top_p < 1.0:
            sorted_logits = np.sort(logits)[::-1]
            cumulative_probs = np.cumsum(np.exp(sorted_logits - np.max(sorted_logits)))
            cumulative_probs = cumulative_probs / cumulative_probs[-1]
            
            # Remove tokens with cumulative probability above the threshold
            mask = cumulative_probs > self.inference_config.top_p
            if np.any(mask):
                threshold = sorted_logits[np.where(mask)[0][0]]
                logits[logits < threshold] = -float('inf')
        
        # Convert to probabilities
        probs = np.exp(logits - np.max(logits))
        probs = probs / np.sum(probs)
        
        # Sample
        next_token = np.random.choice(len(probs), p=probs)
        
        return next_token


def test_engine():
    """Test the inference engine"""
    print("Testing Mini-vLLM Inference Engine...")
    
    # Create configs
    model_config = ModelConfig(
        num_layers=4,
        num_heads=8,
        num_kv_heads=2,
        head_dim=64,
        hidden_dim=512,
        intermediate_dim=1024,
        vocab_size=1000,
        max_seq_len=128,
        dtype="fp32"
    )
    
    inference_config = InferenceConfig(
        max_tokens=512,
        temperature=0.8,
        top_p=0.95,
        top_k=50
    )
    
    # Create engine
    engine = InferenceEngine(model_config, inference_config)
    
    # Mock tokenizer
    class MockTokenizer:
        def encode(self, text):
            return [1, 2, 3, 4, 5]  # Simple mock
        
        @property
        def eos_token_id(self):
            return 0
    
    tokenizer = MockTokenizer()
    
    # Add sequence
    seq_id = engine.add_sequence("Hello, world!", tokenizer)
    print(f"Added sequence with ID: {seq_id}")
    
    # Prefill
    logits = engine.prefill(seq_id)
    print(f"Prefill logits shape: {logits.shape}")
    
    # Generate a few tokens
    generated = engine.generate(seq_id, 5, tokenizer)
    print(f"Generated tokens: {generated}")
    
    print("Test completed!")


if __name__ == "__main__":
    test_engine()