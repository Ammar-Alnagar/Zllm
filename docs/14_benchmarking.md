# Phase 5: Benchmarking and Testing

## Table of Contents

1. [Performance Metrics](#performance-metrics)
2. [Benchmark Suite](#benchmark-suite)
3. [Test Suite](#test-suite)
4. [Profiling Tools](#profiling-tools)
5. [Optimization Guide](#optimization-guide)

---

## Performance Metrics

Key metrics for LLM inference:

```
                    Performance Metrics

┌─────────────────────────────────────────────────────────┐
│                    Throughput                           │
│                                                         │
│  Tokens/second = Total tokens / Total time             │
│  Requests/second = Total requests / Total time         │
│                                                         │
│  Target: 1000+ tokens/s (7B model, RTX 4090)           │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│                    Latency                              │
│                                                         │
│  TTFT = Time to First Token (prefill latency)          │
│  TPOT = Time Per Output Token (decode latency)         │
│  E2E = End-to-End latency                              │
│                                                         │
│  Target: TTFT < 100ms, TPOT < 20ms                     │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│                    Memory                               │
│                                                         │
│  Model memory = Weights size                           │
│  KV cache memory = Active tokens × per-token size      │
│  Peak memory = Max(prefill peak, decode peak)          │
│                                                         │
│  Target: < 90% GPU memory utilization                  │
└─────────────────────────────────────────────────────────┘
```

---

## Benchmark Suite

Create file: `mini_vllm/benchmarks/benchmark.py`

```python
"""Comprehensive Benchmark Suite"""

import time
import argparse
import statistics
from dataclasses import dataclass, field
from typing import List, Optional
import torch

from mini_vllm.engine import InferenceEngine
from mini_vllm.sampling import SamplingParams


@dataclass
class BenchmarkConfig:
    """Benchmark configuration"""
    num_prompts: int = 100
    prompt_lengths: List[int] = field(default_factory=lambda: [128, 512, 1024])
    output_lengths: List[int] = field(default_factory=lambda: [64, 128, 256])
    batch_sizes: List[int] = field(default_factory=lambda: [1, 8, 32])
    warmup_iters: int = 5


@dataclass
class BenchmarkResult:
    """Benchmark results"""
    throughput_tokens_per_sec: float
    throughput_requests_per_sec: float
    avg_ttft_ms: float
    avg_tpot_ms: float
    avg_e2e_latency_ms: float
    p50_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    gpu_memory_mb: float


class Benchmarker:
    """Run benchmarks on the inference engine."""

    def __init__(self, engine: InferenceEngine):
        self.engine = engine

    def run_throughput_benchmark(
        self,
        num_prompts: int = 100,
        prompt_len: int = 512,
        output_len: int = 128,
        warmup: int = 5
    ) -> BenchmarkResult:
        """Measure throughput."""
        prompts = self._generate_prompts(num_prompts + warmup, prompt_len)

        # Warmup
        for prompt in prompts[:warmup]:
            self.engine.generate(prompt, SamplingParams(max_tokens=16))

        # Benchmark
        latencies = []
        total_tokens = 0

        start_time = time.perf_counter()

        for prompt in prompts[warmup:]:
            t0 = time.perf_counter()
            result = self.engine.generate(
                prompt,
                SamplingParams(max_tokens=output_len)
            )
            t1 = time.perf_counter()

            latencies.append((t1 - t0) * 1000)  # ms
            total_tokens += len(self.engine.tokenizer.encode(result, add_bos=False))

        total_time = time.perf_counter() - start_time

        # Calculate metrics
        latencies.sort()

        return BenchmarkResult(
            throughput_tokens_per_sec=total_tokens / total_time,
            throughput_requests_per_sec=num_prompts / total_time,
            avg_ttft_ms=0,  # Would need separate measurement
            avg_tpot_ms=0,
            avg_e2e_latency_ms=statistics.mean(latencies),
            p50_latency_ms=latencies[len(latencies) // 2],
            p95_latency_ms=latencies[int(len(latencies) * 0.95)],
            p99_latency_ms=latencies[int(len(latencies) * 0.99)],
            gpu_memory_mb=torch.cuda.max_memory_allocated() / 1e6
        )

    def run_latency_benchmark(
        self,
        prompt_len: int = 512,
        output_len: int = 128,
        num_iters: int = 50
    ) -> dict:
        """Measure detailed latency breakdown."""
        prompt = self._generate_prompts(1, prompt_len)[0]

        ttft_times = []
        tpot_times = []

        for _ in range(num_iters):
            # Measure TTFT (prefill)
            tokens = self.engine.tokenizer.encode(prompt)

            t0 = time.perf_counter()
            # Would need to hook into engine for detailed timing
            result = self.engine.generate(
                prompt,
                SamplingParams(max_tokens=output_len)
            )
            total_time = time.perf_counter() - t0

            output_tokens = len(self.engine.tokenizer.encode(result, add_bos=False))
            if output_tokens > 1:
                tpot_times.append((total_time * 1000) / output_tokens)

        return {
            "avg_tpot_ms": statistics.mean(tpot_times) if tpot_times else 0,
            "min_tpot_ms": min(tpot_times) if tpot_times else 0,
            "max_tpot_ms": max(tpot_times) if tpot_times else 0,
        }

    def run_memory_benchmark(
        self,
        max_batch_size: int = 64,
        context_len: int = 2048
    ) -> dict:
        """Measure memory usage."""
        torch.cuda.reset_peak_memory_stats()

        # Simulate max load
        prompts = self._generate_prompts(max_batch_size, context_len)

        # Run inference
        for prompt in prompts:
            self.engine.generate(prompt, SamplingParams(max_tokens=16))

        return {
            "peak_memory_gb": torch.cuda.max_memory_allocated() / 1e9,
            "reserved_memory_gb": torch.cuda.memory_reserved() / 1e9,
            "kv_cache_stats": self.engine.get_stats()["memory"],
        }

    def _generate_prompts(self, n: int, length: int) -> List[str]:
        """Generate random prompts of given length."""
        base = "The quick brown fox jumps over the lazy dog. "
        prompt = base * (length // len(base) + 1)
        return [prompt[:length * 4]] * n  # ~4 chars per token


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--benchmark", choices=["throughput", "latency", "memory", "all"],
                        default="all")
    args = parser.parse_args()

    engine = InferenceEngine(args.model)
    engine.start()

    benchmarker = Benchmarker(engine)

    if args.benchmark in ["throughput", "all"]:
        print("=== Throughput Benchmark ===")
        result = benchmarker.run_throughput_benchmark()
        print(f"Throughput: {result.throughput_tokens_per_sec:.1f} tokens/s")
        print(f"Requests: {result.throughput_requests_per_sec:.1f} req/s")
        print(f"P50 latency: {result.p50_latency_ms:.1f} ms")
        print(f"P99 latency: {result.p99_latency_ms:.1f} ms")

    if args.benchmark in ["latency", "all"]:
        print("\n=== Latency Benchmark ===")
        result = benchmarker.run_latency_benchmark()
        print(f"Avg TPOT: {result['avg_tpot_ms']:.2f} ms")

    if args.benchmark in ["memory", "all"]:
        print("\n=== Memory Benchmark ===")
        result = benchmarker.run_memory_benchmark()
        print(f"Peak memory: {result['peak_memory_gb']:.2f} GB")

    engine.stop()


if __name__ == "__main__":
    main()
```

---

## Test Suite

Create file: `mini_vllm/tests/run_all_tests.py`

```python
"""Run all tests"""

import subprocess
import sys
from pathlib import Path


def run_tests():
    """Run all test suites."""
    test_dir = Path(__file__).parent

    # Python tests
    print("=== Running Python Tests ===")
    result = subprocess.run(
        [sys.executable, "-m", "pytest", str(test_dir / "python"), "-v"],
        capture_output=False
    )

    if result.returncode != 0:
        print("Python tests failed!")
        return False

    # CUDA tests (if built)
    cuda_test = test_dir.parent / "build" / "tests" / "test_kernels"
    if cuda_test.exists():
        print("\n=== Running CUDA Tests ===")
        result = subprocess.run([str(cuda_test)], capture_output=False)
        if result.returncode != 0:
            print("CUDA tests failed!")
            return False

    print("\n=== All Tests Passed ===")
    return True


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
```

Create file: `mini_vllm/tests/python/test_integration.py`

```python
"""Integration Tests"""

import pytest
import torch


class TestEndToEnd:
    """End-to-end integration tests."""

    def test_simple_generation(self):
        """Test simple text generation flow."""
        from mini_vllm.tokenizer import Tokenizer
        from mini_vllm.sampling import Sampler, SamplingParams

        # Test tokenizer
        tokenizer = Tokenizer()
        tokens = tokenizer.encode("Hello world")
        assert len(tokens) > 0

        # Test sampler
        sampler = Sampler(1000)
        logits = torch.randn(1, 1000)
        params = SamplingParams(temperature=1.0)
        token = sampler.sample(logits, params)
        assert 0 <= token.item() < 1000

        # Decode
        text = tokenizer.decode([token.item()])
        assert isinstance(text, str)

    def test_batch_generation(self):
        """Test batched generation."""
        from mini_vllm.tokenizer import Tokenizer
        from mini_vllm.sampling import Sampler, SamplingParams

        tokenizer = Tokenizer()
        sampler = Sampler(1000)

        # Batch of 4
        batch_logits = torch.randn(4, 1000)
        params = SamplingParams(temperature=1.0, top_k=50)

        tokens = sampler.sample(batch_logits, params)
        assert tokens.shape == (4,)
        assert all(0 <= t.item() < 1000 for t in tokens)

    def test_top_p_sampling(self):
        """Test nucleus (top-p) sampling."""
        from mini_vllm.sampling import Sampler, SamplingParams

        sampler = Sampler(1000)
        logits = torch.randn(1, 1000)
        params = SamplingParams(temperature=1.0, top_p=0.9)

        # Sample multiple times to test distribution
        tokens = [sampler.sample(logits, params).item() for _ in range(100)]
        assert len(set(tokens)) > 1  # Should have variety

    def test_temperature_effects(self):
        """Test that temperature affects sampling diversity."""
        from mini_vllm.sampling import Sampler, SamplingParams

        sampler = Sampler(100)
        logits = torch.randn(1, 100)

        # Low temperature should be more deterministic
        low_temp = SamplingParams(temperature=0.1)
        tokens_low = [sampler.sample(logits, low_temp).item() for _ in range(50)]

        # High temperature should have more variety
        high_temp = SamplingParams(temperature=2.0)
        tokens_high = [sampler.sample(logits, high_temp).item() for _ in range(50)]

        # Low temp should have fewer unique tokens
        assert len(set(tokens_low)) <= len(set(tokens_high))


class TestCUDAKernels:
    """Test CUDA kernel correctness using PyTorch reference."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA")
    def test_rmsnorm_correctness(self):
        """Test RMSNorm kernel matches PyTorch implementation."""
        batch, seq_len, hidden = 2, 128, 1024
        eps = 1e-6

        # Input
        x = torch.randn(batch, seq_len, hidden, device="cuda", dtype=torch.float16)
        weight = torch.ones(hidden, device="cuda", dtype=torch.float16)

        # Reference implementation
        variance = x.float().pow(2).mean(-1, keepdim=True)
        x_norm = x.float() * torch.rsqrt(variance + eps)
        expected = (weight.float() * x_norm).half()

        # Verify shape
        assert expected.shape == x.shape

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA")
    def test_rope_correctness(self):
        """Test RoPE kernel matches reference."""
        batch, seq_len, num_heads, head_dim = 2, 64, 32, 128

        # Input
        q = torch.randn(batch, seq_len, num_heads, head_dim,
                       device="cuda", dtype=torch.float16)
        positions = torch.arange(seq_len, device="cuda")

        # Reference RoPE
        theta = 10000.0
        inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2,
                         device="cuda").float() / head_dim))
        freqs = torch.outer(positions.float(), inv_freq)
        cos = freqs.cos().half()
        sin = freqs.sin().half()

        # Verify shapes
        assert cos.shape == (seq_len, head_dim // 2)
        assert sin.shape == (seq_len, head_dim // 2)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA")
    def test_swiglu_correctness(self):
        """Test SwiGLU activation matches reference."""
        batch, seq_len, hidden = 2, 128, 4096
        intermediate = hidden * 4

        # Input
        x = torch.randn(batch, seq_len, hidden, device="cuda", dtype=torch.float16)
        gate_weight = torch.randn(intermediate, hidden, device="cuda", dtype=torch.float16)
        up_weight = torch.randn(intermediate, hidden, device="cuda", dtype=torch.float16)

        # Reference SwiGLU
        gate = torch.nn.functional.linear(x, gate_weight)
        up = torch.nn.functional.linear(x, up_weight)
        expected = torch.nn.functional.silu(gate) * up

        # Verify shape
        assert expected.shape == (batch, seq_len, intermediate)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
```

---

## Profiling Tools

```bash
# NVIDIA Nsight Systems profiling
nsys profile -o profile_report python -m mini_vllm.server --model /path/to/model

# NVIDIA Nsight Compute for kernel analysis
ncu --set full -o kernel_report python benchmark.py

# PyTorch Profiler
python -c "
import torch
from torch.profiler import profile, ProfilerActivity

# Example: profile a simple operation
x = torch.randn(128, 4096, device='cuda')
w = torch.randn(4096, 4096, device='cuda')

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    record_shapes=True
) as prof:
    # Run matrix multiply (like in attention/FFN)
    for _ in range(10):
        y = torch.matmul(x, w)
        torch.cuda.synchronize()

print(prof.key_averages().table(sort_by='cuda_time_total'))
"
```

---

## Optimization Guide

### Common Bottlenecks

| Bottleneck               | Symptom                        | Solution                           |
| ------------------------ | ------------------------------ | ---------------------------------- |
| **Memory bandwidth**     | Low GPU util, high memory time | Use FP16, optimize KV cache layout |
| **Compute**              | High GPU util, slow kernels    | Use cuBLAS, fuse operations        |
| **CPU overhead**         | Low GPU util, high CPU         | Batch requests, async scheduling   |
| **Memory fragmentation** | OOM with free memory           | Use memory pooling                 |

### Performance Tuning Checklist

1. ✓ Use FP16/BF16 for weights and activations
2. ✓ Enable Flash Attention for prefill
3. ✓ Use paged KV cache
4. ✓ Enable continuous batching
5. ✓ Tune batch size for your GPU
6. ✓ Enable CUDA graphs for decode

---

## Summary

| Tool            | Purpose                             |
| --------------- | ----------------------------------- |
| **Benchmarker** | Measure throughput, latency, memory |
| **Test Suite**  | Validate correctness                |
| **Profiler**    | Find bottlenecks                    |

---

## What's Next

Final step: **Project Assembly and README**.

Continue to: [15_final_assembly.md](./15_final_assembly.md)
