# Zllm: Build Your Own LLM Inference Engine

A hands-on educational project for building a high-performance LLM inference engine from scratch, targeting the **Qwen3** architecture.

## What You'll Build

By following the documentation in `docs/`, you'll implement:

- **Custom CUDA Kernels**: RMSNorm, RoPE, SwiGLU, Flash Attention
- **Paged KV Cache**: Memory-efficient caching with 16-token blocks
- **RadixAttention**: Prefix sharing across requests using a radix tree
- **Continuous Batching**: Dynamic request scheduling
- **FastAPI Server**: OpenAI-compatible REST API

## Project Structure

```
Zllm/
├── README.md                   # This file
├── docs/                       #  DOCUMENTATION (START HERE!)
│   ├── 00_introduction.md      # Overview & prerequisites
│   ├── 01_project_setup.md     # Setup & CMake
│   ├── 02_cuda_foundations.md  # CUDA basics
│   ├── 03_rmsnorm_kernel.md    # RMSNorm implementation
│   ├── 04_rope_kernel.md       # RoPE implementation
│   ├── 05_swiglu_kernel.md     # SwiGLU implementation
│   ├── 06_flash_attention_prefill.md
│   ├── 07_flash_infer_decode.md
│   ├── 08_paged_kv_cache.md
│   ├── 09_radix_attention.md
│   ├── 10_memory_pool.md
│   ├── 11_scheduler.md
│   ├── 12_model_runner.md
│   ├── 13_inference_engine.md
│   ├── 14_benchmarking.md
│   └── 15_final_assembly.md
│
└── zllm/                       # Implementation (fill using docs/)
    ├── CMakeLists.txt          # Root CMake (docs/01)
    ├── setup.py                # Python build (docs/01)
    ├── pyproject.toml          # Python config (docs/01)
    │
    ├── csrc/                   # C++/CUDA Source
    │   ├── CMakeLists.txt
    │   ├── include/            # Headers (docs/01)
    │   │   ├── common.cuh
    │   │   ├── cuda_utils.cuh
    │   │   └── types.hpp
    │   ├── kernels/            # CUDA Kernels
    │   │   ├── rmsnorm.cu/cuh  # (docs/03)
    │   │   ├── rope.cu/cuh     # (docs/04)
    │   │   └── swiglu.cu/cuh   # (docs/05)
    │   ├── attention/          # Attention (docs/06-07)
    │   │   ├── flash_attention.*
    │   │   └── flash_infer.*
    │   ├── memory/             # Memory (docs/08-10)
    │   │   ├── block_manager.hpp
    │   │   ├── kv_cache.*
    │   │   ├── radix_tree.hpp
    │   │   └── memory_pool.cuh
    │   └── bindings/           # pybind11 (docs/01)
    │       └── bindings.cpp
    │
    ├── python/                 # Python Package
    │   ├── __init__.py
    │   ├── tokenizer.py        # (docs/12)
    │   ├── sampling.py         # (docs/12)
    │   ├── scheduler.py        # (docs/11)
    │   ├── model_loader.py     # (docs/12)
    │   ├── model_runner.py     # (docs/12)
    │   ├── kv_cache.py         # (docs/08)
    │   ├── radix_cache.py      # (docs/09)
    │   ├── memory.py           # (docs/10)
    │   ├── engine.py           # (docs/13)
    │   └── server.py           # (docs/13)
    │
    ├── tests/                  # Tests
    │   ├── cpp/test_kernels.cu
    │   └── python/test_*.py
    │
    └── benchmarks/             # Benchmarks (docs/14)
        └── benchmark.py
```

## Getting Started

### Prerequisites

| Requirement | Version                                 |
| ----------- | --------------------------------------- |
| **GPU**     | NVIDIA RTX 3080+ / A100+ (Compute 8.0+) |
| **CUDA**    | 12.1+                                   |
| **Python**  | 3.10+                                   |
| **PyTorch** | 2.1+                                    |
| **CMake**   | 3.24+                                   |
| **GCC**     | 11.0+                                   |

### Quick Start

```bash
# 1. Clone the repository
git clone https://github.com/your-username/Zllm.git
cd Zllm

# 2. Create Python environment
python -m venv .venv
source .venv/bin/activate

# 3. Install dependencies
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install tiktoken fastapi uvicorn pybind11 safetensors pytest

# 4. Start learning!
# Open docs/00_introduction.md and follow along
```

## How to Use This Project

1. **Read `docs/00_introduction.md`** - Understand the architecture
2. **Follow each doc sequentially** - Type out the code yourself
3. **Fill in the empty files in `zllm/`** - Implementation guides are in `docs/`
4. **Test as you go** - Each doc includes test code
5. **Run benchmarks** - Validate your implementation

## Documentation Guide

| Phase               | Docs  | What You'll Learn                           |
| ------------------- | ----- | ------------------------------------------- |
| **0: Setup**        | 00-01 | Prerequisites, project structure, CMake     |
| **1: CUDA Kernels** | 02-07 | RMSNorm, RoPE, SwiGLU, Flash Attention      |
| **2: Memory**       | 08-10 | Paged KV cache, RadixAttention, memory pool |
| **3: Python**       | 11-12 | Scheduler, model runner, tokenizer          |
| **4: Server**       | 13    | Inference engine, FastAPI                   |
| **5: Testing**      | 14-15 | Benchmarks, final assembly                  |

## Learning Goals

After completing this project, you'll understand:

- GPU memory hierarchy and optimization
- CUDA kernel development for ML
- Flash Attention algorithm
- Paged memory management
- Prefix caching with radix trees
- Continuous batching for throughput
- Production inference server design

## Target Performance

| Metric     | Target (7B model, RTX 4090) |
| ---------- | --------------------------- |
| Throughput | ~1000 tokens/s              |
| TTFT       | <100ms                      |
| TPOT       | <20ms                       |
| Memory     | <90% utilization            |

## Tech Stack

- **C++17** - Core implementation
- **CUDA 12.1** - GPU kernels
- **cuBLAS** - Matrix operations
- **Python 3.10** - High-level API
- **PyTorch 2.1** - Tensor operations
- **tiktoken** - Tokenization
- **FastAPI** - REST server
- **pybind11** - Python bindings

## License

MIT License - Feel free to use for learning!

## Acknowledgments

Inspired by:

- [vLLM](https://github.com/vllm-project/vllm)
- [SGLang](https://github.com/sgl-project/sglang)
- [FlashAttention](https://github.com/Dao-AILab/flash-attention)
- [FlashInfer](https://github.com/flashinfer-ai/flashinfer)

---

**Happy Learning! Start with `docs/00_introduction.md`**


_____
**future**

16_quantization.md          # INT8/FP8 support
17_multi_gpu.md             # Tensor parallelism
18_speculative_decoding.md  # Draft models
19_custom_sampling.md       # Beam search, etc.
