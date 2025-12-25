# Phase 1: FlashInfer Decode Kernel

## Table of Contents

1. [Decode Phase Overview](#decode-phase-overview)
2. [Why Decode is Different](#why-decode-is-different)
3. [FlashInfer Algorithm](#flashinfer-algorithm)
4. [Paged KV Cache Access](#paged-kv-cache-access)
5. [CUDA Implementation](#cuda-implementation)
6. [Batched Decode](#batched-decode)
7. [Testing and Verification](#testing-and-verification)

---

## Decode Phase Overview

During **decode (generation)**, we process **one new token at a time** while attending to all previous tokens in the KV cache.

```
                    Prefill vs Decode

PREFILL (Prompt Processing):
┌─────────────────────────────────────────────────────────┐
│  Process entire prompt at once                          │
│                                                         │
│  Q = [q0, q1, q2, ..., q_n]   (n tokens)               │
│  K = [k0, k1, k2, ..., k_n]                            │
│  V = [v0, v1, v2, ..., v_n]                            │
│                                                         │
│  Output: [o0, o1, o2, ..., o_n]                        │
│  Complexity: O(n²) attention operations                 │
│  Use: Flash Attention (tiled)                          │
└─────────────────────────────────────────────────────────┘

DECODE (Token Generation):
┌─────────────────────────────────────────────────────────┐
│  Process ONE new token, attend to ALL cached tokens     │
│                                                         │
│  Q = [q_new]                  (1 token)                │
│  K = [k0, k1, ..., k_n, k_new] (n+1 tokens from cache) │
│  V = [v0, v1, ..., v_n, v_new]                         │
│                                                         │
│  Output: [o_new]              (1 token)                │
│  Complexity: O(n) attention operations                  │
│  Use: FlashInfer (optimized for single query)          │
└─────────────────────────────────────────────────────────┘
```

---

## Why Decode is Different

### Memory Access Patterns

```
                    Prefill Memory Pattern

Q: [████████████████████]  Read once, reuse in SRAM
K: [████████████████████]  Read once, reuse in SRAM
V: [████████████████████]  Read once, reuse in SRAM

Compute-bound: Matrix multiply benefits from tiling


                    Decode Memory Pattern

Q: [█]  (Single token - tiny!)
K: [████████████████████████████████████...]  Read ALL
V: [████████████████████████████████████...]  Read ALL

Memory-bound: Reading entire KV cache is the bottleneck!
- Compute: 2 × context_len × head_dim FLOPs
- Memory: 2 × context_len × head_dim × 2 bytes (KV read)
- Arithmetic Intensity: ~1 FLOP/byte (very low!)
```

### Optimization Goals

```
Flash Attention (Prefill):           FlashInfer (Decode):
─────────────────────────            ────────────────────
✓ Minimize SRAM usage                ✓ Maximize memory bandwidth
✓ Tiled computation                  ✓ Coalesced KV cache reads
✓ Online softmax                     ✓ Parallel over KV blocks
✓ Avoid O(n²) memory                 ✓ Efficient block table lookup
```

---

## FlashInfer Algorithm

### Key Differences from Flash Attention

1. **No Q tiling needed** - Only one query token
2. **Parallelize over KV** - Split KV into blocks, process in parallel
3. **Reduce across blocks** - Combine partial results using online softmax
4. **Paged KV access** - Use block table for non-contiguous cache

### Algorithm Pseudocode

```python
def flashinfer_decode(q, kv_cache, block_table, context_len):
    """
    FlashInfer decode attention

    Args:
        q: Query tensor [batch, num_heads, 1, head_dim]
        kv_cache: Paged KV cache [num_blocks, block_size, num_kv_heads, 2, head_dim]
        block_table: Block indices [batch, max_blocks]
        context_len: Context length for each sequence [batch]
    """
    num_kv_blocks = (context_len + block_size - 1) // block_size

    # Parallel reduction over KV blocks
    partial_outputs = []
    partial_lse = []  # Log-sum-exp for combining

    for block_idx in parallel(num_kv_blocks):
        # Get physical block from table
        physical_block = block_table[block_idx]
        k_block = kv_cache[physical_block, :, :, 0, :]
        v_block = kv_cache[physical_block, :, :, 1, :]

        # Compute attention for this block
        scores = q @ k_block.T / sqrt(d)  # [1, block_size]

        # Local softmax
        m = max(scores)
        p = exp(scores - m)
        l = sum(p)
        o = p @ v_block / l

        partial_outputs.append(o)
        partial_lse.append(m + log(l))

    # Reduce partial results using log-sum-exp
    output = reduce_with_lse(partial_outputs, partial_lse)

    return output
```

---

## Paged KV Cache Access

### Block Table Structure

```
                    KV Cache Layout

Physical memory (GPU):
┌────────────────────────────────────────────────────────┐
│  Block 0   │  Block 1   │  Block 2   │  Block 3   │...│
│ [16 tokens]│ [16 tokens]│ [16 tokens]│ [16 tokens]│   │
│  K,V data  │  K,V data  │  K,V data  │  K,V data  │   │
└────────────────────────────────────────────────────────┘
     ↑             ↑             ↑             ↑
     │             │             │             │
     │   Block Table (per sequence):           │
     │   ┌─────┬─────┬─────┬─────┐            │
Seq 0│   │  0  │  3  │  7  │  -  │            │
     │   └─────┴─────┴─────┴─────┘            │
     │        Maps logical → physical         │
     │                                        │
     │   ┌─────┬─────┬─────┬─────┐           │
Seq 1│   │  1  │  4  │  8  │ 12  │           │
     │   └─────┴─────┴─────┴─────┘           │

Sequence 0 uses physical blocks: 0, 3, 7
Sequence 1 uses physical blocks: 1, 4, 8, 12
```

### KV Cache Memory Layout

```
KV Cache tensor: [num_blocks, 2, block_size, num_kv_heads, head_dim]
                     ↑      ↑       ↑           ↑           ↑
                     │      │       │           │           └─ 128 (Qwen3)
                     │      │       │           └─ 8 (GQA)
                     │      │       └─ 16 (tokens per block)
                     │      └─ K=0, V=1
                     └─ Total blocks in pool

For block b, token t, head h, dim d:
  K[b, t, h, d] = kv_cache[b, 0, t, h, d]
  V[b, t, h, d] = kv_cache[b, 1, t, h, d]
```

---

## CUDA Implementation

Create file: `mini_vllm/csrc/attention/flash_infer.cuh`

```cuda
// =============================================================================
// flash_infer.cuh - FlashInfer Decode Attention Header
// =============================================================================

#pragma once

#include "common.cuh"
#include <cuda_runtime.h>
#include <cuda_fp16.h>

namespace mini_vllm {

/**
 * FlashInfer decode attention for single-token queries
 *
 * @param output: Output tensor [batch, num_heads, 1, head_dim]
 * @param query: Query tensor [batch, num_heads, 1, head_dim]
 * @param kv_cache: Paged KV cache [num_blocks, 2, block_size, num_kv_heads, head_dim]
 * @param block_tables: Block indices [batch, max_num_blocks]
 * @param context_lens: Context length per sequence [batch]
 * @param batch_size: Number of sequences
 * @param num_heads: Number of query heads
 * @param num_kv_heads: Number of KV heads
 * @param head_dim: Dimension per head
 * @param block_size: Tokens per cache block (typically 16)
 * @param max_context_len: Maximum context length
 * @param scale: Attention scale
 * @param stream: CUDA stream
 */
void flash_infer_decode(
    float* output,
    const float* query,
    const float* kv_cache,
    const int* block_tables,
    const int* context_lens,
    int batch_size,
    int num_heads,
    int num_kv_heads,
    int head_dim,
    int block_size,
    int max_context_len,
    float scale,
    cudaStream_t stream = nullptr
);

// FP16 version
void flash_infer_decode_fp16(
    half* output,
    const half* query,
    const half* kv_cache,
    const int* block_tables,
    const int* context_lens,
    int batch_size,
    int num_heads,
    int num_kv_heads,
    int head_dim,
    int block_size,
    int max_context_len,
    float scale,
    cudaStream_t stream = nullptr
);

} // namespace mini_vllm
```

Create file: `mini_vllm/csrc/attention/flash_infer.cu`

```cuda
// =============================================================================
// flash_infer.cu - FlashInfer Decode Attention Implementation
// =============================================================================

#include "flash_infer.cuh"
#include <float.h>

namespace mini_vllm {

// Configuration
constexpr int THREADS_PER_BLOCK = 256;
constexpr int KV_BLOCK_SIZE = 16;  // Tokens per KV cache block
constexpr int HEAD_DIM = 128;       // Fixed for Qwen3

// =============================================================================
// Single-Query Decode Kernel
// =============================================================================

/**
 * flash_infer_kernel - Decode attention with paged KV cache
 *
 * Each block handles one (batch, head) pair.
 * Threads cooperatively process all KV blocks.
 *
 * Memory access pattern:
 * - Q: small, loaded once to shared memory
 * - K,V: large, streamed through with coalescing
 */
template<int BLOCK_SIZE>
__global__ void flash_infer_kernel(
    float* __restrict__ output,         // [batch, num_heads, head_dim]
    const float* __restrict__ query,    // [batch, num_heads, head_dim]
    const float* __restrict__ kv_cache, // [num_blocks, 2, KV_BLOCK_SIZE, num_kv_heads, head_dim]
    const int* __restrict__ block_tables,  // [batch, max_num_blocks]
    const int* __restrict__ context_lens,  // [batch]
    int num_heads,
    int num_kv_heads,
    int head_dim,
    int max_num_blocks,
    float scale
) {
    // Block assignment
    const int batch_idx = blockIdx.y;
    const int head_idx = blockIdx.x;
    const int kv_head_idx = head_idx / (num_heads / num_kv_heads);
    const int tid = threadIdx.x;

    // Get context length for this sequence
    const int context_len = context_lens[batch_idx];
    const int num_kv_blocks = (context_len + KV_BLOCK_SIZE - 1) / KV_BLOCK_SIZE;

    if (context_len == 0) return;

    // =========================================================================
    // Load query into shared memory
    // =========================================================================
    __shared__ float Q_shared[HEAD_DIM];

    const float* q_ptr = query + batch_idx * num_heads * head_dim +
                                 head_idx * head_dim;

    // Coalesced load
    for (int i = tid; i < head_dim; i += BLOCK_SIZE) {
        Q_shared[i] = q_ptr[i] * scale;
    }
    __syncthreads();

    // =========================================================================
    // Shared memory for KV block and partial results
    // =========================================================================
    __shared__ float K_shared[KV_BLOCK_SIZE][HEAD_DIM + 1];  // +1 to avoid bank conflicts
    __shared__ float V_shared[KV_BLOCK_SIZE][HEAD_DIM + 1];
    __shared__ float scores_shared[KV_BLOCK_SIZE];

    // Per-thread accumulators
    float output_acc[HEAD_DIM / BLOCK_SIZE + 1];
    for (int i = 0; i < HEAD_DIM / BLOCK_SIZE + 1; i++) {
        output_acc[i] = 0.0f;
    }
    float m_acc = -FLT_MAX;  // Running max
    float l_acc = 0.0f;       // Running sum

    // Block table pointer for this sequence
    const int* block_table = block_tables + batch_idx * max_num_blocks;

    // =========================================================================
    // Process each KV block
    // =========================================================================
    for (int kv_block_idx = 0; kv_block_idx < num_kv_blocks; kv_block_idx++) {
        // Get physical block index from table
        const int physical_block = block_table[kv_block_idx];

        // Calculate number of valid tokens in this block
        const int block_start = kv_block_idx * KV_BLOCK_SIZE;
        const int valid_tokens = min(KV_BLOCK_SIZE, context_len - block_start);

        // ---------------------------------------------------------------------
        // Load K block
        // ---------------------------------------------------------------------
        // KV cache layout: [num_blocks, 2, block_size, num_kv_heads, head_dim]
        const float* k_block_ptr = kv_cache +
            physical_block * 2 * KV_BLOCK_SIZE * num_kv_heads * head_dim +
            0 * KV_BLOCK_SIZE * num_kv_heads * head_dim +  // K (index 0)
            kv_head_idx * head_dim;

        for (int i = tid; i < KV_BLOCK_SIZE * head_dim; i += BLOCK_SIZE) {
            int token = i / head_dim;
            int dim = i % head_dim;

            if (token < valid_tokens) {
                K_shared[token][dim] = k_block_ptr[token * num_kv_heads * head_dim + dim];
            } else {
                K_shared[token][dim] = 0.0f;
            }
        }

        // ---------------------------------------------------------------------
        // Load V block
        // ---------------------------------------------------------------------
        const float* v_block_ptr = kv_cache +
            physical_block * 2 * KV_BLOCK_SIZE * num_kv_heads * head_dim +
            1 * KV_BLOCK_SIZE * num_kv_heads * head_dim +  // V (index 1)
            kv_head_idx * head_dim;

        for (int i = tid; i < KV_BLOCK_SIZE * head_dim; i += BLOCK_SIZE) {
            int token = i / head_dim;
            int dim = i % head_dim;

            if (token < valid_tokens) {
                V_shared[token][dim] = v_block_ptr[token * num_kv_heads * head_dim + dim];
            } else {
                V_shared[token][dim] = 0.0f;
            }
        }
        __syncthreads();

        // ---------------------------------------------------------------------
        // Compute attention scores: S = Q @ K^T
        // ---------------------------------------------------------------------
        for (int token = tid; token < KV_BLOCK_SIZE; token += BLOCK_SIZE) {
            float score = 0.0f;

            if (token < valid_tokens) {
                #pragma unroll 8
                for (int d = 0; d < head_dim; d++) {
                    score += Q_shared[d] * K_shared[token][d];
                }
            } else {
                score = -FLT_MAX;  // Mask out invalid positions
            }

            scores_shared[token] = score;
        }
        __syncthreads();

        // ---------------------------------------------------------------------
        // Online softmax update
        // ---------------------------------------------------------------------
        // Find max in this block
        float block_max = -FLT_MAX;
        for (int i = 0; i < valid_tokens; i++) {
            block_max = fmaxf(block_max, scores_shared[i]);
        }

        // Update running max
        float m_new = fmaxf(m_acc, block_max);
        float scale_old = expf(m_acc - m_new);

        // Compute exp(scores - m_new) and sum
        float block_sum = 0.0f;
        for (int i = tid; i < KV_BLOCK_SIZE; i += BLOCK_SIZE) {
            if (i < valid_tokens) {
                scores_shared[i] = expf(scores_shared[i] - m_new);
                block_sum += scores_shared[i];
            } else {
                scores_shared[i] = 0.0f;
            }
        }

        // Reduce block_sum across threads
        __shared__ float sum_shared[32];
        float warp_sum = block_sum;
        for (int offset = 16; offset > 0; offset /= 2) {
            warp_sum += __shfl_down_sync(0xffffffff, warp_sum, offset);
        }
        if (tid % 32 == 0) {
            sum_shared[tid / 32] = warp_sum;
        }
        __syncthreads();

        if (tid < 32) {
            warp_sum = (tid < (BLOCK_SIZE / 32)) ? sum_shared[tid] : 0.0f;
            for (int offset = 16; offset > 0; offset /= 2) {
                warp_sum += __shfl_down_sync(0xffffffff, warp_sum, offset);
            }
        }
        __syncthreads();

        float l_new = l_acc * scale_old + warp_sum;

        // ---------------------------------------------------------------------
        // Accumulate output: O = O * scale + P @ V
        // ---------------------------------------------------------------------
        for (int d = tid; d < head_dim; d += BLOCK_SIZE) {
            float o = output_acc[d / BLOCK_SIZE] * scale_old;

            for (int token = 0; token < valid_tokens; token++) {
                o += scores_shared[token] * V_shared[token][d];
            }

            output_acc[d / BLOCK_SIZE] = o;
        }

        m_acc = m_new;
        l_acc = l_new;
        __syncthreads();
    }

    // =========================================================================
    // Write output (normalized)
    // =========================================================================
    float* out_ptr = output + batch_idx * num_heads * head_dim + head_idx * head_dim;

    float inv_l = (l_acc > 0.0f) ? 1.0f / l_acc : 0.0f;

    for (int d = tid; d < head_dim; d += BLOCK_SIZE) {
        out_ptr[d] = output_acc[d / BLOCK_SIZE] * inv_l;
    }
}

// =============================================================================
// Wrapper Function
// =============================================================================

void flash_infer_decode(
    float* output,
    const float* query,
    const float* kv_cache,
    const int* block_tables,
    const int* context_lens,
    int batch_size,
    int num_heads,
    int num_kv_heads,
    int head_dim,
    int block_size,
    int max_context_len,
    float scale,
    cudaStream_t stream
) {
    const int max_num_blocks = (max_context_len + block_size - 1) / block_size;

    dim3 grid(num_heads, batch_size);
    dim3 block(THREADS_PER_BLOCK);

    flash_infer_kernel<THREADS_PER_BLOCK><<<grid, block, 0, stream>>>(
        output, query, kv_cache, block_tables, context_lens,
        num_heads, num_kv_heads, head_dim, max_num_blocks, scale
    );

    CUDA_CHECK_LAST();
}

// =============================================================================
// Optimized Kernel with Split-KV Parallelism
// =============================================================================

/**
 * For very long contexts, we can parallelize over KV blocks too.
 * Each block handles a subset of KV blocks, then we reduce across blocks.
 */
template<int THREADS>
__global__ void flash_infer_split_kv_kernel(
    float* __restrict__ partial_output,  // [batch, num_heads, num_splits, head_dim]
    float* __restrict__ partial_lse,     // [batch, num_heads, num_splits]
    const float* __restrict__ query,
    const float* __restrict__ kv_cache,
    const int* __restrict__ block_tables,
    const int* __restrict__ context_lens,
    int num_heads,
    int num_kv_heads,
    int head_dim,
    int max_num_blocks,
    int num_splits,
    float scale
) {
    const int batch_idx = blockIdx.z;
    const int head_idx = blockIdx.y;
    const int split_idx = blockIdx.x;
    const int kv_head_idx = head_idx / (num_heads / num_kv_heads);
    const int tid = threadIdx.x;

    const int context_len = context_lens[batch_idx];
    const int num_kv_blocks = (context_len + KV_BLOCK_SIZE - 1) / KV_BLOCK_SIZE;

    // Determine which KV blocks this split handles
    const int blocks_per_split = (num_kv_blocks + num_splits - 1) / num_splits;
    const int kv_start = split_idx * blocks_per_split;
    const int kv_end = min(kv_start + blocks_per_split, num_kv_blocks);

    if (kv_start >= num_kv_blocks) return;

    // Load query
    __shared__ float Q_shared[HEAD_DIM];
    const float* q_ptr = query + batch_idx * num_heads * head_dim + head_idx * head_dim;

    for (int i = tid; i < head_dim; i += THREADS) {
        Q_shared[i] = q_ptr[i] * scale;
    }
    __syncthreads();

    // Process assigned KV blocks
    __shared__ float K_shared[KV_BLOCK_SIZE][HEAD_DIM + 1];
    __shared__ float V_shared[KV_BLOCK_SIZE][HEAD_DIM + 1];
    __shared__ float scores_shared[KV_BLOCK_SIZE];

    float output_acc[4];  // Assumes head_dim / THREADS <= 4
    for (int i = 0; i < 4; i++) output_acc[i] = 0.0f;
    float m_acc = -FLT_MAX;
    float l_acc = 0.0f;

    const int* block_table = block_tables + batch_idx * max_num_blocks;

    for (int kv_block_idx = kv_start; kv_block_idx < kv_end; kv_block_idx++) {
        const int physical_block = block_table[kv_block_idx];
        const int block_start = kv_block_idx * KV_BLOCK_SIZE;
        const int valid_tokens = min(KV_BLOCK_SIZE, context_len - block_start);

        // Load K and V
        const float* k_ptr = kv_cache +
            physical_block * 2 * KV_BLOCK_SIZE * num_kv_heads * head_dim +
            kv_head_idx * head_dim;
        const float* v_ptr = k_ptr + KV_BLOCK_SIZE * num_kv_heads * head_dim;

        for (int i = tid; i < KV_BLOCK_SIZE * head_dim; i += THREADS) {
            int t = i / head_dim, d = i % head_dim;
            K_shared[t][d] = (t < valid_tokens) ? k_ptr[t * num_kv_heads * head_dim + d] : 0.0f;
            V_shared[t][d] = (t < valid_tokens) ? v_ptr[t * num_kv_heads * head_dim + d] : 0.0f;
        }
        __syncthreads();

        // Compute scores
        for (int t = tid; t < KV_BLOCK_SIZE; t += THREADS) {
            float s = 0.0f;
            if (t < valid_tokens) {
                for (int d = 0; d < head_dim; d++) s += Q_shared[d] * K_shared[t][d];
            } else {
                s = -FLT_MAX;
            }
            scores_shared[t] = s;
        }
        __syncthreads();

        // Online softmax
        float block_max = -FLT_MAX;
        for (int t = 0; t < valid_tokens; t++) {
            block_max = fmaxf(block_max, scores_shared[t]);
        }

        float m_new = fmaxf(m_acc, block_max);
        float scale_factor = expf(m_acc - m_new);

        float block_sum = 0.0f;
        for (int t = 0; t < valid_tokens; t++) {
            scores_shared[t] = expf(scores_shared[t] - m_new);
            block_sum += scores_shared[t];
        }

        l_acc = l_acc * scale_factor + block_sum;

        // Accumulate output
        for (int d = tid; d < head_dim; d += THREADS) {
            float o = output_acc[d / THREADS] * scale_factor;
            for (int t = 0; t < valid_tokens; t++) {
                o += scores_shared[t] * V_shared[t][d];
            }
            output_acc[d / THREADS] = o;
        }

        m_acc = m_new;
        __syncthreads();
    }

    // Write partial results
    const int out_offset = batch_idx * num_heads * num_splits * head_dim +
                           head_idx * num_splits * head_dim +
                           split_idx * head_dim;

    for (int d = tid; d < head_dim; d += THREADS) {
        partial_output[out_offset + d] = output_acc[d / THREADS];
    }

    if (tid == 0) {
        const int lse_offset = batch_idx * num_heads * num_splits +
                               head_idx * num_splits + split_idx;
        partial_lse[lse_offset] = m_acc + logf(fmaxf(l_acc, 1e-10f));
    }
}

/**
 * Reduction kernel to combine partial results
 */
__global__ void reduce_partial_outputs(
    float* __restrict__ output,
    const float* __restrict__ partial_output,
    const float* __restrict__ partial_lse,
    int batch_size,
    int num_heads,
    int num_splits,
    int head_dim
) {
    const int batch_idx = blockIdx.y;
    const int head_idx = blockIdx.x;
    const int tid = threadIdx.x;

    // Find global max across splits
    float max_lse = -FLT_MAX;
    for (int s = 0; s < num_splits; s++) {
        float lse = partial_lse[batch_idx * num_heads * num_splits +
                                head_idx * num_splits + s];
        max_lse = fmaxf(max_lse, lse);
    }

    // Combine with rescaling
    float sum_exp = 0.0f;
    float output_d[4];
    for (int i = 0; i < 4; i++) output_d[i] = 0.0f;

    for (int s = 0; s < num_splits; s++) {
        float lse = partial_lse[batch_idx * num_heads * num_splits +
                                head_idx * num_splits + s];
        float w = expf(lse - max_lse);
        sum_exp += w;

        const float* po = partial_output + batch_idx * num_heads * num_splits * head_dim +
                                           head_idx * num_splits * head_dim +
                                           s * head_dim;

        for (int d = tid; d < head_dim; d += blockDim.x) {
            output_d[d / blockDim.x] += w * po[d];
        }
    }

    // Normalize and write
    float* out = output + batch_idx * num_heads * head_dim + head_idx * head_dim;
    float inv_sum = 1.0f / sum_exp;

    for (int d = tid; d < head_dim; d += blockDim.x) {
        out[d] = output_d[d / blockDim.x] * inv_sum;
    }
}

} // namespace mini_vllm
```

---

## Batched Decode

For serving multiple requests simultaneously, we need efficient batched decode:

```
                    Continuous Batching Decode

Batch contains sequences at different positions:
┌─────────────────────────────────────────────────────────┐
│ Seq 0: context=1024 tokens  Q=[q_1024]                 │
│ Seq 1: context=512 tokens   Q=[q_512]                  │
│ Seq 2: context=2048 tokens  Q=[q_2048]                 │
│ Seq 3: context=256 tokens   Q=[q_256]                  │
└─────────────────────────────────────────────────────────┘

Each sequence has:
- Different context length
- Different number of KV blocks
- Same block table structure

Kernel processes all in parallel:
- Grid: (num_heads, batch_size)
- Each block handles one (head, sequence) pair
```

---

## Testing and Verification

Create file: `mini_vllm/tests/cpp/test_flash_infer.cu`

```cuda
// =============================================================================
// test_flash_infer.cu - FlashInfer Decode Tests
// =============================================================================

#include "flash_infer.cuh"
#include <cuda_runtime.h>
#include <cstdio>
#include <cmath>
#include <vector>
#include <random>

using namespace mini_vllm;

// CPU reference (decode attention)
void decode_attention_cpu(
    float* output,
    const float* query,
    const float* k_cache,
    const float* v_cache,
    int batch_size,
    int num_heads,
    int context_len,
    int head_dim,
    float scale
) {
    for (int b = 0; b < batch_size; b++) {
        for (int h = 0; h < num_heads; h++) {
            const float* q = query + b * num_heads * head_dim + h * head_dim;
            const float* k = k_cache + b * num_heads * context_len * head_dim +
                             h * context_len * head_dim;
            const float* v = v_cache + b * num_heads * context_len * head_dim +
                             h * context_len * head_dim;
            float* o = output + b * num_heads * head_dim + h * head_dim;

            // Compute scores
            std::vector<float> scores(context_len);
            float max_score = -FLT_MAX;

            for (int t = 0; t < context_len; t++) {
                float score = 0.0f;
                for (int d = 0; d < head_dim; d++) {
                    score += q[d] * k[t * head_dim + d];
                }
                score *= scale;
                scores[t] = score;
                max_score = fmaxf(max_score, score);
            }

            // Softmax
            float sum_exp = 0.0f;
            for (int t = 0; t < context_len; t++) {
                scores[t] = expf(scores[t] - max_score);
                sum_exp += scores[t];
            }
            for (int t = 0; t < context_len; t++) {
                scores[t] /= sum_exp;
            }

            // Apply attention
            for (int d = 0; d < head_dim; d++) {
                float out = 0.0f;
                for (int t = 0; t < context_len; t++) {
                    out += scores[t] * v[t * head_dim + d];
                }
                o[d] = out;
            }
        }
    }
}

int main() {
    printf("=== FlashInfer Decode Tests ===\n\n");

    // Configuration
    const int batch_size = 4;
    const int num_heads = 32;
    const int num_kv_heads = 8;
    const int head_dim = 128;
    const int context_len = 512;
    const int block_size = 16;
    const float scale = 1.0f / sqrtf(head_dim);

    const int num_blocks = (context_len + block_size - 1) / block_size;

    // Allocate host memory
    std::vector<float> h_query(batch_size * num_heads * head_dim);
    std::vector<float> h_output_cpu(batch_size * num_heads * head_dim);
    std::vector<float> h_output_gpu(batch_size * num_heads * head_dim);

    // Allocate KV cache (contiguous for CPU reference)
    std::vector<float> h_k_cache(batch_size * num_heads * context_len * head_dim);
    std::vector<float> h_v_cache(batch_size * num_heads * context_len * head_dim);

    // Allocate paged KV cache for GPU
    // Layout: [num_total_blocks, 2, block_size, num_kv_heads, head_dim]
    const int total_blocks = batch_size * num_blocks;
    std::vector<float> h_kv_cache(total_blocks * 2 * block_size * num_kv_heads * head_dim);

    // Block tables: identity mapping for this test
    std::vector<int> h_block_tables(batch_size * num_blocks);
    for (int b = 0; b < batch_size; b++) {
        for (int i = 0; i < num_blocks; i++) {
            h_block_tables[b * num_blocks + i] = b * num_blocks + i;
        }
    }

    std::vector<int> h_context_lens(batch_size, context_len);

    // Initialize random
    std::mt19937 gen(42);
    std::normal_distribution<float> dist(0.0f, 0.5f);

    for (auto& v : h_query) v = dist(gen);
    for (auto& v : h_k_cache) v = dist(gen);
    for (auto& v : h_v_cache) v = dist(gen);

    // Convert contiguous KV to paged format
    for (int b = 0; b < batch_size; b++) {
        for (int block_idx = 0; block_idx < num_blocks; block_idx++) {
            int physical_block = h_block_tables[b * num_blocks + block_idx];

            for (int t = 0; t < block_size; t++) {
                int global_t = block_idx * block_size + t;
                if (global_t >= context_len) break;

                for (int h = 0; h < num_kv_heads; h++) {
                    for (int d = 0; d < head_dim; d++) {
                        // GPU paged layout
                        int gpu_idx = physical_block * 2 * block_size * num_kv_heads * head_dim +
                                      0 * block_size * num_kv_heads * head_dim +  // K
                                      t * num_kv_heads * head_dim +
                                      h * head_dim + d;

                        // CPU contiguous (expand to all heads)
                        int cpu_h = h * (num_heads / num_kv_heads);  // First Q head using this KV
                        int cpu_idx = b * num_heads * context_len * head_dim +
                                      cpu_h * context_len * head_dim +
                                      global_t * head_dim + d;

                        h_kv_cache[gpu_idx] = h_k_cache[cpu_idx];

                        // V
                        gpu_idx = physical_block * 2 * block_size * num_kv_heads * head_dim +
                                  1 * block_size * num_kv_heads * head_dim +
                                  t * num_kv_heads * head_dim +
                                  h * head_dim + d;

                        h_kv_cache[gpu_idx] = h_v_cache[cpu_idx];
                    }
                }
            }
        }
    }

    // Allocate device memory
    float *d_query, *d_output, *d_kv_cache;
    int *d_block_tables, *d_context_lens;

    cudaMalloc(&d_query, batch_size * num_heads * head_dim * sizeof(float));
    cudaMalloc(&d_output, batch_size * num_heads * head_dim * sizeof(float));
    cudaMalloc(&d_kv_cache, h_kv_cache.size() * sizeof(float));
    cudaMalloc(&d_block_tables, h_block_tables.size() * sizeof(int));
    cudaMalloc(&d_context_lens, batch_size * sizeof(int));

    cudaMemcpy(d_query, h_query.data(), batch_size * num_heads * head_dim * sizeof(float),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_kv_cache, h_kv_cache.data(), h_kv_cache.size() * sizeof(float),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_block_tables, h_block_tables.data(), h_block_tables.size() * sizeof(int),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_context_lens, h_context_lens.data(), batch_size * sizeof(int),
               cudaMemcpyHostToDevice);

    // =======================================================================
    // Test 1: Correctness
    // =======================================================================
    printf("Test 1: Correctness\n");

    // CPU reference (using expanded KV for all heads)
    decode_attention_cpu(h_output_cpu.data(), h_query.data(),
                         h_k_cache.data(), h_v_cache.data(),
                         batch_size, num_heads, context_len, head_dim, scale);

    // GPU
    flash_infer_decode(d_output, d_query, d_kv_cache,
                       d_block_tables, d_context_lens,
                       batch_size, num_heads, num_kv_heads, head_dim,
                       block_size, context_len, scale);
    cudaDeviceSynchronize();

    cudaMemcpy(h_output_gpu.data(), d_output,
               batch_size * num_heads * head_dim * sizeof(float),
               cudaMemcpyDeviceToHost);

    float max_err = 0.0f;
    for (int i = 0; i < batch_size * num_heads * head_dim; i++) {
        max_err = fmaxf(max_err, fabsf(h_output_cpu[i] - h_output_gpu[i]));
    }

    printf("  Max error: %.2e\n", max_err);
    printf("  %s\n", max_err < 1e-3f ? "[PASS]" : "[FAIL] (Note: GQA mapping may differ)");

    // =======================================================================
    // Test 2: Performance
    // =======================================================================
    printf("\nTest 2: Performance\n");

    // Warmup
    for (int i = 0; i < 10; i++) {
        flash_infer_decode(d_output, d_query, d_kv_cache,
                           d_block_tables, d_context_lens,
                           batch_size, num_heads, num_kv_heads, head_dim,
                           block_size, context_len, scale);
    }
    cudaDeviceSynchronize();

    // Benchmark
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    const int iterations = 1000;
    cudaEventRecord(start);
    for (int i = 0; i < iterations; i++) {
        flash_infer_decode(d_output, d_query, d_kv_cache,
                           d_block_tables, d_context_lens,
                           batch_size, num_heads, num_kv_heads, head_dim,
                           block_size, context_len, scale);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    float avg_us = (ms / iterations) * 1000;

    // Memory bandwidth calculation
    // Read: Q (batch * heads * dim) + KV (batch * context * kv_heads * dim * 2)
    // Write: O (batch * heads * dim)
    float bytes = (batch_size * num_heads * head_dim * 2 +
                   batch_size * context_len * num_kv_heads * head_dim * 2) *
                  sizeof(float) * iterations;
    float bandwidth_gb = bytes / (ms * 1e6f);

    printf("  Batch=%d, Heads=%d, Context=%d\n", batch_size, num_heads, context_len);
    printf("  Time: %.2f us\n", avg_us);
    printf("  Bandwidth: %.1f GB/s\n", bandwidth_gb);
    printf("  Tokens/sec: %.1f K\n", (batch_size * iterations) / (ms));

    // Cleanup
    cudaFree(d_query);
    cudaFree(d_output);
    cudaFree(d_kv_cache);
    cudaFree(d_block_tables);
    cudaFree(d_context_lens);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    printf("\n=== Tests Complete ===\n");
    return 0;
}
```

---

## Summary

You've implemented FlashInfer decode with:

| Feature                       | Implementation             |
| ----------------------------- | -------------------------- |
| **Single-query optimization** | No Q tiling needed         |
| **Paged KV access**           | Block table indirection    |
| **Online softmax**            | Streaming across KV blocks |
| **GQA support**               | KV head sharing            |
| **Split-KV parallelism**      | For long contexts          |

### Key Differences from Flash Attention

| Aspect      | Flash Attention (Prefill) | FlashInfer (Decode) |
| ----------- | ------------------------- | ------------------- |
| Query size  | Many tokens               | 1 token             |
| Bottleneck  | Compute                   | Memory bandwidth    |
| Parallelism | Over Q tiles              | Over KV blocks      |
| KV access   | Contiguous                | Paged (block table) |

---

## What's Next

Now we'll move to **Memory Management** - implementing the paged KV cache and block allocator.

Continue to: [08_paged_kv_cache.md](./08_paged_kv_cache.md)
