# Phase 0: Introduction to Mini-vLLM

## Table of Contents

1. [Project Overview](#project-overview)
2. [What We're Building](#what-were-building)
3. [Architecture Overview](#architecture-overview)
4. [Prerequisites](#prerequisites)
5. [Hardware Requirements](#hardware-requirements)
6. [Software Dependencies](#software-dependencies)
7. [Key Concepts](#key-concepts)

---

## Project Overview

This guide walks you through building a **mini-vLLM** inference engine from scratch. vLLM is a high-throughput, memory-efficient inference engine for Large Language Models. Our implementation targets the **Qwen3 architecture**, which uses modern transformer techniques for efficient inference.

By the end of this guide, you will have built:

- Custom CUDA kernels for attention, normalization, and activations
- A paged KV cache with RadixAttention for prefix sharing
- A continuous batching scheduler
- A FastAPI server for deployment

### Why Build This?

Understanding LLM inference at the kernel level gives you:

1. **Deep understanding** of GPU memory hierarchy and optimization
2. **Practical skills** in CUDA programming for ML
3. **Knowledge of production systems** like vLLM, SGLang, and TensorRT-LLM
4. **Ability to optimize** inference for your specific use case

---

## What We're Building

Our mini-vLLM implements these core components:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         Mini-vLLM Architecture                          │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌─────────────┐    ┌──────────────┐    ┌─────────────────────────────┐ │
│  │   FastAPI   │───▶│   Scheduler  │───▶│     Inference Engine        │ │
│  │   Server    │    │  (Continuous │    │  ┌─────────────────────────┐│ │
│  │             │    │   Batching)  │    │  │   Transformer Layers    ││ │
│  └─────────────┘    └──────────────┘    │  │  ┌───────────────────┐  ││ │
│                                          │  │  │    Attention      │  ││ │
│  ┌─────────────┐    ┌──────────────┐    │  │  │  (Flash/FlashInfer)│  ││ │
│  │  Tokenizer  │    │   Sampler    │    │  │  └───────────────────┘  ││ │
│  │  (tiktoken) │    │ (top-p/top-k)│    │  │  ┌───────────────────┐  ││ │
│  └─────────────┘    └──────────────┘    │  │  │   FFN (SwiGLU)    │  ││ │
│                                          │  │  └───────────────────┘  ││ │
│  ┌──────────────────────────────────┐   │  │  ┌───────────────────┐  ││ │
│  │         Memory Manager           │   │  │  │    RMSNorm        │  ││ │
│  │  ┌────────────┐ ┌─────────────┐  │   │  │  └───────────────────┘  ││ │
│  │  │ Block      │ │   Radix     │  │   │  │  ┌───────────────────┐  ││ │
│  │  │ Allocator  │ │   Tree      │  │   │  │  │     RoPE          │  ││ │
│  │  └────────────┘ └─────────────┘  │   │  │  └───────────────────┘  ││ │
│  │  ┌────────────────────────────┐  │   │  └─────────────────────────┘│ │
│  │  │    Paged KV Cache          │  │   └─────────────────────────────┘ │
│  │  │    (16-token blocks)       │  │                                   │
│  │  └────────────────────────────┘  │                                   │
│  └──────────────────────────────────┘                                   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### Target Model: Qwen3 Architecture

Qwen3 uses these key components that we'll implement:

| Component      | Description                                             | Our Implementation |
| -------------- | ------------------------------------------------------- | ------------------ |
| **GQA**        | Grouped Query Attention - fewer KV heads than Q heads   | Custom CUDA kernel |
| **RoPE**       | Rotary Position Embeddings - position info via rotation | CUDA kernel        |
| **SwiGLU**     | Swish-Gated Linear Unit activation                      | CUDA kernel        |
| **RMSNorm**    | Root Mean Square Normalization                          | CUDA kernel        |
| **Vocabulary** | ~150K tokens using tiktoken                             | Python integration |

---

## Architecture Overview

### Data Flow During Inference

```
                              PREFILL PHASE
                    ┌─────────────────────────────────┐
                    │                                 │
   Input Tokens     │   Process entire prompt at      │    KV Cache
   [T1, T2, ... Tn]─┼─▶ once using Flash Attention   ─┼──▶ Populated
                    │   Tiled computation in SRAM     │
                    │                                 │
                    └─────────────────────────────────┘
                                    │
                                    ▼
                              DECODE PHASE
                    ┌─────────────────────────────────┐
                    │                                 │
   Previous Token   │   Generate one token at a time  │    Next Token
   [Tn]────────────┼─▶ using FlashInfer (optimized   ─┼──▶ [Tn+1]
                    │   for single-token attention)   │
                    │                                 │
                    └─────────────────────────────────┘
                                    │
                                    ▼
                              ┌───────────┐
                              │  Sampler  │
                              │ (top-p/k) │
                              └───────────┘
                                    │
                                    ▼
                              Output Token
```

### Memory Layout: Paged KV Cache

```
GPU Memory Pool (Pre-allocated)
┌────────────────────────────────────────────────────────────────────┐
│                                                                    │
│   Block 0        Block 1        Block 2        Block 3       ...   │
│  ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐        │
│  │ 16 tokens│   │ 16 tokens│   │ 16 tokens│   │ 16 tokens│        │
│  │ K,V pairs│   │ K,V pairs│   │ K,V pairs│   │ K,V pairs│        │
│  │          │   │          │   │          │   │          │        │
│  │ Layer 0  │   │ Layer 0  │   │ Layer 0  │   │ Layer 0  │        │
│  │ Layer 1  │   │ Layer 1  │   │ Layer 1  │   │ Layer 1  │        │
│  │   ...    │   │   ...    │   │   ...    │   │   ...    │        │
│  │ Layer N  │   │ Layer N  │   │ Layer N  │   │ Layer N  │        │
│  └──────────┘   └──────────┘   └──────────┘   └──────────┘        │
│                                                                    │
└────────────────────────────────────────────────────────────────────┘

Block Table (per sequence):
┌──────────┬──────────┬──────────┬──────────┬───┐
│ Seq 0    │ Block 0  │ Block 3  │ Block 7  │...│
├──────────┼──────────┼──────────┼──────────┼───┤
│ Seq 1    │ Block 1  │ Block 4  │ Block 8  │...│
├──────────┼──────────┼──────────┼──────────┼───┤
│ Seq 2    │ Block 2  │ Block 5  │ Block 9  │...│
└──────────┴──────────┴──────────┴──────────┴───┘
```

### RadixAttention: Prefix Sharing

```
                         Radix Tree for Prefix Sharing

                                  ┌─────────┐
                                  │  Root   │
                                  └────┬────┘
                                       │
                         ┌─────────────┼────────────────┐
                         │             │                │
                         ▼             ▼                ▼
                   ┌──────────┐  ┌──────────┐    ┌──────────┐
                   │"System:" │  │ "User:"  │    │"Assist:" │
                   │ Block 0  │  │ Block 5  │    │ Block 10 │
                   │ ref=2    │  │ ref=1    │    │ ref=1    │
                   └────┬─────┘  └──────────┘    └──────────┘
                        │
              ┌─────────┴─────────┐
              │                   │
              ▼                   ▼
        ┌──────────┐        ┌──────────┐
        │"You are" │        │"Answer:" │
        │ Block 1  │        │ Block 3  │
        │ ref=2    │        │ ref=1    │
        └────┬─────┘        └──────────┘
             │
    ┌────────┴────────┐
    │                 │
    ▼                 ▼
┌──────────┐    ┌──────────┐
│"helpful" │    │"coding"  │
│ Block 2  │    │ Block 4  │
│ ref=1    │    │ ref=1    │
└──────────┘    └──────────┘

Requests sharing "System: You are" prefix share Blocks 0 and 1
```

---

## Prerequisites

### Knowledge Requirements

Before starting, you should be comfortable with:

1. **C++ (Intermediate)**

   - Templates and modern C++ (C++17)
   - Memory management
   - Build systems (CMake)

2. **CUDA (Beginner to Intermediate)**

   - Basic kernel writing
   - Thread/block organization
   - We'll teach memory optimization

3. **Python (Intermediate)**

   - PyTorch tensor operations
   - Async programming (asyncio)
   - Web frameworks (FastAPI)

4. **Machine Learning**
   - Transformer architecture basics
   - Attention mechanism understanding
   - Tokenization concepts

---

## Hardware Requirements

### Minimum Requirements

| Component        | Minimum                | Recommended            |
| ---------------- | ---------------------- | ---------------------- |
| **GPU**          | NVIDIA RTX 3080 (10GB) | NVIDIA RTX 4090 (24GB) |
| **CUDA Compute** | 8.0 (Ampere)           | 8.9 (Ada Lovelace)     |
| **GPU Memory**   | 10 GB                  | 24+ GB                 |
| **System RAM**   | 32 GB                  | 64 GB                  |
| **Storage**      | 50 GB SSD              | 100 GB NVMe            |

### Supported GPU Architectures

```
┌─────────────────────────────────────────────────────────────────┐
│                    Supported NVIDIA GPUs                        │
├─────────────────────────────────────────────────────────────────┤
│  Architecture    │  Compute Cap.  │  Example GPUs               │
├──────────────────┼────────────────┼─────────────────────────────┤
│  Ampere          │  8.0, 8.6      │  A100, RTX 3080/3090        │
│  Ada Lovelace    │  8.9           │  RTX 4080/4090, L40         │
│  Hopper          │  9.0           │  H100, H200                 │
└─────────────────────────────────────────────────────────────────┘
```

### Check Your GPU

Run this command to verify your GPU:

```bash
nvidia-smi --query-gpu=name,compute_cap,memory.total --format=csv
```

Expected output:

```
name, compute_cap, memory.total [MiB]
NVIDIA GeForce RTX 4090, 8.9, 24564 MiB
```

---

## Software Dependencies

### System Requirements

| Software         | Version     | Purpose                |
| ---------------- | ----------- | ---------------------- |
| **Ubuntu**       | 22.04 LTS   | Operating system       |
| **CUDA Toolkit** | 12.1+       | GPU programming        |
| **cuBLAS**       | (with CUDA) | Matrix multiplication  |
| **GCC**          | 11.0+       | C++ compiler           |
| **CMake**        | 3.24+       | Build system           |
| **Python**       | 3.10+       | High-level integration |
| **PyTorch**      | 2.1+        | Tensor operations      |

### Installation Commands

#### 1. CUDA Toolkit

```bash
# Download and install CUDA 12.1
wget https://developer.download.nvidia.com/compute/cuda/12.1.0/local_installers/cuda_12.1.0_530.30.02_linux.run
sudo sh cuda_12.1.0_530.30.02_linux.run

# Add to PATH (add to ~/.bashrc)
export PATH=/usr/local/cuda-12.1/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.1/lib64:$LD_LIBRARY_PATH

# Verify installation
nvcc --version
```

#### 2. Build Tools

```bash
# Install GCC and CMake
sudo apt update
sudo apt install -y build-essential cmake ninja-build

# Verify versions
gcc --version    # Should be 11.0+
cmake --version  # Should be 3.24+
```

#### 3. Python Environment

```bash
# Create virtual environment
python3 -m venv mini_vllm_env
source mini_vllm_env/bin/activate

# Install Python dependencies
pip install --upgrade pip
pip install torch==2.1.0 --index-url https://download.pytorch.org/whl/cu121
pip install tiktoken==0.5.1
pip install fastapi==0.104.1
pip install uvicorn==0.24.0
pip install pybind11==2.11.1
pip install numpy==1.24.0
pip install pytest==7.4.3
```

#### 4. Verify PyTorch CUDA

```python
# Run this Python script to verify
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
print(f"GPU: {torch.cuda.get_device_name(0)}")
```

Expected output:

```
PyTorch version: 2.1.0+cu121
CUDA available: True
CUDA version: 12.1
GPU: NVIDIA GeForce RTX 4090
```

---

## Key Concepts

Before diving into implementation, understand these concepts:

### 1. Transformer Decoder Architecture

```
                    Transformer Decoder Block (Qwen3)

Input ──────────────────────────────────────────────────────▶
   │                                                         │
   ▼                                                         │
┌─────────────────────────────────────────────────────────┐  │
│                      RMSNorm                            │  │
└────────────────────────────┬────────────────────────────┘  │
                             │                               │
                             ▼                               │
┌─────────────────────────────────────────────────────────┐  │
│              Grouped Query Attention (GQA)              │  │
│  ┌─────────┐    ┌─────────┐    ┌─────────┐             │  │
│  │    Q    │    │    K    │    │    V    │             │  │
│  │ n_heads │    │n_kv_head│    │n_kv_head│             │  │
│  └─────────┘    └─────────┘    └─────────┘             │  │
│       │              │              │                   │  │
│       └──────────────┴──────────────┘                   │  │
│                      │                                  │  │
│              ┌───────┴───────┐                          │  │
│              │  Attention    │                          │  │
│              │  (Flash/Infer)│                          │  │
│              └───────────────┘                          │  │
└────────────────────────────┬────────────────────────────┘  │
                             │                               │
                             ├───────────────────────────────┤
                             │  (Residual Connection)        │
                             ▼                               │
┌─────────────────────────────────────────────────────────┐  │
│                      RMSNorm                            │  │
└────────────────────────────┬────────────────────────────┘  │
                             │                               │
                             ▼                               │
┌─────────────────────────────────────────────────────────┐  │
│                   FFN (SwiGLU)                          │  │
│  ┌──────────┐   ┌─────────┐   ┌───────────────────┐    │  │
│  │ Linear 1 │──▶│  SiLU   │──▶│                   │    │  │
│  │ (gate)   │   │ (Swish) │   │   Element-wise ×  │    │  │
│  └──────────┘   └─────────┘   │                   │    │  │
│  ┌──────────┐                 │                   │    │  │
│  │ Linear 2 │─────────────────┤                   │    │  │
│  │  (up)    │                 │                   │    │  │
│  └──────────┘                 └─────────┬─────────┘    │  │
│                                         │              │  │
│                               ┌─────────▼─────────┐    │  │
│                               │     Linear 3      │    │  │
│                               │      (down)       │    │  │
│                               └───────────────────┘    │  │
└────────────────────────────┬────────────────────────────┘  │
                             │                               │
                             ├───────────────────────────────┘
                             │  (Residual Connection)
                             ▼
                          Output
```

### 2. Grouped Query Attention (GQA)

GQA reduces memory by sharing Key-Value heads across multiple Query heads:

```
Multi-Head Attention (MHA):        Grouped Query Attention (GQA):
   32 Q heads                         32 Q heads
   32 K heads                          8 K heads (shared)
   32 V heads                          8 V heads (shared)

   Q1 K1 V1                           Q1-Q4   K1 V1
   Q2 K2 V2                           Q5-Q8   K2 V2
   Q3 K3 V3                           Q9-Q12  K3 V3
   ...                                ...

   Memory: 3 × 32 × d                 Memory: 32 + 8 + 8 = 48 × d
                                      (vs 96 × d for MHA)
```

### 3. RoPE (Rotary Position Embeddings)

RoPE encodes position by rotating query and key vectors:

```
Position encoding via rotation in 2D subspaces:

For position m and dimension pair (2i, 2i+1):

┌        ┐   ┌                    ┐   ┌    ┐
│ q'_2i  │ = │ cos(mθ)  -sin(mθ) │ × │ q_2i│
│ q'_2i+1│   │ sin(mθ)   cos(mθ) │   │q_2i+1│
└        ┘   └                    ┘   └    ┘

Where θ = 10000^(-2i/d)

This allows relative position information:
q'_m · k'_n = f(q, k, m-n)  ← Only depends on relative position!
```

### 4. Flash Attention

Flash Attention minimizes memory I/O by computing attention in tiles:

```
Traditional Attention:              Flash Attention:

Load Q, K, V (HBM → SRAM)          Load Q tile (HBM → SRAM)
Compute S = Q @ K^T                 For each K, V tile:
Write S to HBM                        Load K, V tile
Load S from HBM                       Compute partial attention
Compute P = softmax(S)                Update running softmax
Write P to HBM                        Accumulate output
Load P from HBM                     Write final output
Compute O = P @ V
Write O to HBM

Memory I/O: O(N²d)                  Memory I/O: O(N²d²/M)
                                    M = SRAM size
```

### 5. Continuous Batching

Unlike static batching, continuous batching adds/removes sequences dynamically:

```
Time →
          Static Batching                 Continuous Batching

Step 1:   [Seq1████  ]  ← Wait          [Seq1████]  ← Start
          [Seq2██████]     for          [Seq2██████]    generating
          [Seq3████  ]     all          [Seq3████]

Step 2:   [Seq1████  ]  ← All           [Seq1████] ← Seq1 done
          [Seq2██████]     wait         [Seq2████ ]
          [Seq3████  ]     for          [Seq3██  ]
                           Seq2         [Seq4█████] ← New seq joins!

Step 3:   [Seq1████  ]                  [Seq2████] ← Seq2 done
          [Seq2██████] ← Finally        [Seq3███ ]
          [Seq3████  ]   generate       [Seq4████]
                                        [Seq5██  ] ← New seq joins!
```

---

## What's Next

In the next document, we'll set up the project structure:

1. Create the directory hierarchy
2. Set up CMake build system
3. Configure Python bindings
4. Create the development environment

Continue to: [01_project_setup.md](./01_project_setup.md)

---

## Quick Reference

### File Extensions We'll Use

| Extension | Purpose                  |
| --------- | ------------------------ |
| `.cu`     | CUDA kernel source files |
| `.cuh`    | CUDA header files        |
| `.cpp`    | C++ source files         |
| `.hpp`    | C++ header files         |
| `.py`     | Python source files      |

### Directory Structure Preview

```
mini_vllm/
├── csrc/                    # C++/CUDA source
│   ├── kernels/             # CUDA kernels
│   ├── attention/           # Attention implementations
│   └── memory/              # Memory management
├── python/                  # Python package
│   ├── mini_vllm/           # Main package
│   ├── tests/               # Python tests
│   └── setup.py             # Package setup
├── CMakeLists.txt           # Build configuration
└── README.md                # Project readme
```
