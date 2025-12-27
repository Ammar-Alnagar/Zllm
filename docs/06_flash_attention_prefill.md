# Phase 1: Flash Attention Prefill Kernel

## Table of Contents

1. [Understanding Flash Attention](#understanding-flash-attention)
2. [Algorithm Overview](#algorithm-overview)
3. [Online Softmax](#online-softmax)
4. [Tiled Computation](#tiled-computation)
5. [CUDA Implementation](#cuda-implementation)
6. [GQA Support](#gqa-support)
7. [Testing and Verification](#testing-and-verification)

---

## Understanding Flash Attention

Flash Attention computes exact attention while dramatically reducing memory I/O. This is critical for LLM inference where attention is the main bottleneck.

```
                    Why Flash Attention?

Standard Attention Memory Complexity:
┌─────────────────────────────────────────────────────────┐
│  1. S = Q @ K^T        → O(N²) memory for scores       │
│  2. P = softmax(S)     → O(N²) memory for probs        │
│  3. O = P @ V          → O(N×d) memory for output      │
│                                                         │
│  Total: Must store N×N matrices in HBM                  │
│  For N=4096, d=128: 4096² × 4 bytes = 64 MB per head!  │
└─────────────────────────────────────────────────────────┘

Flash Attention Memory Complexity:
┌─────────────────────────────────────────────────────────┐
│  1. Load Q, K, V tiles into SRAM (fast)                │
│  2. Compute attention in tiles                          │
│  3. Never materialize full N×N matrix!                  │
│                                                         │
│  Total: O(N × d²/M) HBM accesses (M = SRAM size)       │
│  Typically 2-4x faster than standard attention!         │
└─────────────────────────────────────────────────────────┘
```

### Memory Hierarchy Recap

```
                    GPU Memory Access Costs

┌──────────────────────────────────────────────────────────┐
│                                                          │
│   SRAM (Shared Memory)     HBM (Global Memory)           │
│   ───────────────────      ──────────────────            │
│   19 TB/s bandwidth        1-3 TB/s bandwidth            │
│   48-228 KB per SM         16-80 GB total                │
│   ~20 cycles latency       ~400 cycles latency           │
│                                                          │
│   Flash Attention Strategy:                              │
│   ─────────────────────────                              │
│   Keep Q, K, V tiles in SRAM                             │
│   Compute attention tile by tile                         │
│   Only write final output to HBM                         │
│                                                          │
└──────────────────────────────────────────────────────────┘
```

---

## Algorithm Overview

### Standard Attention (Slow)

```python
def standard_attention(Q, K, V):
    """Standard attention - O(N²) memory"""
    # Q, K, V: [batch, num_heads, seq_len, head_dim]

    # Step 1: Compute attention scores
    S = Q @ K.T  # [batch, heads, N, N] - N² memory!

    # Step 2: Scale and softmax
    S = S / sqrt(head_dim)
    P = softmax(S, dim=-1)  # Another N² memory!

    # Step 3: Apply attention
    O = P @ V  # [batch, heads, N, d]

    return O
```

### Flash Attention (Fast)

```python
def flash_attention(Q, K, V, block_size=64):
    """Flash Attention - O(N×d²/M) HBM access"""
    N, d = Q.shape[-2], Q.shape[-1]

    # Output and running statistics
    O = zeros_like(Q)
    l = zeros(N)  # Running sum of exp(scores)
    m = full(N, -inf)  # Running max of scores

    # Process K, V in blocks
    for j in range(0, N, block_size):
        K_block = K[j:j+block_size]
        V_block = V[j:j+block_size]

        # Process Q in blocks
        for i in range(0, N, block_size):
            Q_block = Q[i:i+block_size]

            # Compute attention for this tile
            S_tile = Q_block @ K_block.T / sqrt(d)

            # Online softmax update
            m_new = max(m[i:i+block_size], S_tile.max(dim=-1))
            P_tile = exp(S_tile - m_new)
            l_new = exp(m - m_new) * l + P_tile.sum(dim=-1)

            # Update output
            O[i] = exp(m - m_new) * O[i] + P_tile @ V_block

            m = m_new
            l = l_new

    # Final normalization
    O = O / l
    return O
```

---

## Online Softmax

The key innovation in Flash Attention is **online softmax**, which computes softmax without materializing the full score matrix.

```
                    Online Softmax Algorithm

Traditional Softmax (requires all scores):
─────────────────────────────────────────
1. Compute all scores: S[i,j] for all j
2. Find max: m = max(S[i,:])
3. Compute exp: e[j] = exp(S[i,j] - m)
4. Compute sum: l = Σ e[j]
5. Normalize: P[i,j] = e[j] / l


Online Softmax (streaming):
──────────────────────────
Process one block at a time, maintaining running statistics:

For each block k:
  1. Compute scores for block: S_k
  2. Update max: m_new = max(m_old, max(S_k))
  3. Rescale previous: scale = exp(m_old - m_new)
  4. Update sum: l_new = l_old × scale + Σ exp(S_k - m_new)
  5. Update output: O_new = O_old × scale + P_k × V_k

Final: O = O / l
```

### Mathematical Derivation

```
Softmax definition:
  softmax(x)_i = exp(x_i) / Σ_j exp(x_j)

For numerical stability, subtract max:
  softmax(x)_i = exp(x_i - m) / Σ_j exp(x_j - m)
  where m = max(x)

Online update for combining two blocks A and B:
  m_combined = max(m_A, m_B)

  l_combined = exp(m_A - m_combined) × l_A + exp(m_B - m_combined) × l_B

  O_combined = (exp(m_A - m_combined) × l_A × O_A +
                exp(m_B - m_combined) × l_B × O_B) / l_combined

Simplified for streaming (O already weighted by l):
  O_new = O_old × exp(m_old - m_new) + softmax(S_new) @ V_new
  l_new = l_old × exp(m_old - m_new) + sum(exp(S_new - m_new))
```

---

## Tiled Computation

### Tile Sizes

```
                    Tile Size Selection

SRAM Budget per SM: 48 KB (conservative) to 228 KB (Hopper)

For each tile we need:
- Q tile: Br × d × 4 bytes (FP32) or 2 bytes (FP16)
- K tile: Bc × d × 4 bytes
- V tile: Bc × d × 4 bytes
- S tile: Br × Bc × 4 bytes
- O tile: Br × d × 4 bytes

Typical tile sizes (FP16, d=128):
─────────────────────────────────
Br (query rows)    = 64
Bc (KV rows)       = 64
d (head dimension) = 128

Memory per tile:
- Q: 64 × 128 × 2 = 16 KB
- K: 64 × 128 × 2 = 16 KB
- V: 64 × 128 × 2 = 16 KB
- S: 64 × 64 × 4  = 16 KB (FP32 accumulator)
- O: 64 × 128 × 4 = 32 KB (FP32 accumulator)

Total: ~96 KB per tile (fits in Ampere+ SRAM)
```

### Tiling Diagram

```
            K^T (transposed, columns = sequence positions)
            ┌───┬───┬───┬───┬───┐
            │K0 │K1 │K2 │K3 │K4 │  ← Bc columns per tile
            │   │   │   │   │   │
            └───┴───┴───┴───┴───┘
              ↑   ↑   ↑   ↑   ↑
    Q         │   │   │   │   │
    ┌───┐     │   │   │   │   │
    │Q0 │─────┼───┼───┼───┼───┤           Output
    ├───┤     │   │   │   │   │           ┌───┐
    │Q1 │─────┼───┼───┼───┼───┤           │O0 │
    ├───┤     │   │   │   │   │           ├───┤
    │Q2 │     │   │   │   │   │           │O1 │
    ├───┤     │   │   │   │   │           ├───┤
    │Q3 │     │   │   │   │   │           │O2 │
    ├───┤     │   │   │   │   │           ├───┤
    │Q4 │     │   │   │   │   │           │O3 │
    └───┘     │   │   │   │   │           ├───┤
      ↑       │   │   │   │   │           │O4 │
    Br rows   │   │   │   │   │           └───┘
              │   │   │   │   │
              ▼   ▼   ▼   ▼   ▼

    Each Q tile computes attention with ALL K tiles
    Output is accumulated using online softmax
```

---

## CUDA Implementation

Create file: `mini_vllm/csrc/attention/flash_attention.cuh`

```c++
// =============================================================================
// flash_attention.cuh - Flash Attention Header
// =============================================================================

#pragma once

#include "common.cuh"
#include <cuda_runtime.h>
#include <cuda_fp16.h>

namespace mini_vllm {

/**
 * Flash Attention forward pass for prefill
 *
 * Computes: O = softmax(Q @ K^T / sqrt(d)) @ V
 *
 * @param output: Output tensor [batch, num_heads, seq_len, head_dim]
 * @param query: Query tensor [batch, num_heads, seq_len, head_dim]
 * @param key: Key tensor [batch, num_kv_heads, seq_len, head_dim]
 * @param value: Value tensor [batch, num_kv_heads, seq_len, head_dim]
 * @param batch_size: Batch size
 * @param num_heads: Number of query heads
 * @param num_kv_heads: Number of KV heads (for GQA)
 * @param seq_len: Sequence length
 * @param head_dim: Dimension per head
 * @param scale: Attention scale (usually 1/sqrt(head_dim))
 * @param is_causal: Whether to apply causal mask
 * @param stream: CUDA stream
 */
void flash_attention_forward(
    float* output,
    const float* query,
    const float* key,
    const float* value,
    int batch_size,
    int num_heads,
    int num_kv_heads,
    int seq_len,
    int head_dim,
    float scale,
    bool is_causal,
    cudaStream_t stream = nullptr
);

// FP16 version
void flash_attention_forward_fp16(
    half* output,
    const half* query,
    const half* key,
    const half* value,
    int batch_size,
    int num_heads,
    int num_kv_heads,
    int seq_len,
    int head_dim,
    float scale,
    bool is_causal,
    cudaStream_t stream = nullptr
);

/**
 * Flash Attention with variable sequence lengths
 *
 * For batches where sequences have different lengths.
 * Uses cu_seqlens format (cumulative sequence lengths).
 *
 * @param output: Output tensor [total_tokens, num_heads, head_dim]
 * @param query: Query tensor [total_tokens, num_heads, head_dim]
 * @param key: Key tensor [total_tokens, num_kv_heads, head_dim]
 * @param value: Value tensor [total_tokens, num_kv_heads, head_dim]
 * @param cu_seqlens_q: Cumulative query sequence lengths [batch_size + 1]
 * @param cu_seqlens_k: Cumulative key sequence lengths [batch_size + 1]
 * @param max_seqlen_q: Maximum query sequence length
 * @param max_seqlen_k: Maximum key sequence length
 * @param batch_size: Number of sequences
 * @param num_heads: Number of query heads
 * @param num_kv_heads: Number of KV heads
 * @param head_dim: Dimension per head
 * @param scale: Attention scale
 * @param is_causal: Whether to apply causal mask
 * @param stream: CUDA stream
 */
void flash_attention_varlen_forward(
    float* output,
    const float* query,
    const float* key,
    const float* value,
    const int* cu_seqlens_q,
    const int* cu_seqlens_k,
    int max_seqlen_q,
    int max_seqlen_k,
    int batch_size,
    int num_heads,
    int num_kv_heads,
    int head_dim,
    float scale,
    bool is_causal,
    cudaStream_t stream = nullptr
);

} // namespace mini_vllm
```

Create file: `mini_vllm/csrc/attention/flash_attention.cu`

```c++
// =============================================================================
// flash_attention.cu - Flash Attention Implementation
// =============================================================================
//
// This is a simplified but functional Flash Attention implementation.
// Production implementations use more aggressive optimizations like:
// - Tensor Cores (WMMA or MMA PTX)
// - Software pipelining
// - More sophisticated tile sizes
// =============================================================================

#include "flash_attention.cuh"
#include <float.h>

namespace mini_vllm {

// Tile sizes - chosen for Ampere GPUs with 48KB shared memory
constexpr int Br = 64;   // Query tile rows
constexpr int Bc = 64;   // KV tile rows
constexpr int d = 128;   // Head dimension (fixed for Qwen3)

// =============================================================================
// Online Softmax Utilities
// =============================================================================

/**
 * Update running max within a warp
 */
__device__ __forceinline__ float warp_max(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
    }
    return __shfl_sync(0xffffffff, val, 0);
}

/**
 * Update running sum within a warp
 */
__device__ __forceinline__ float warp_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return __shfl_sync(0xffffffff, val, 0);
}

// =============================================================================
// Flash Attention Kernel
// =============================================================================

/**
 * flash_attention_kernel - Core Flash Attention computation
 *
 * Each block handles one (batch, head) pair and processes all Q tiles.
 * Within a block, threads cooperatively compute attention.
 *
 * Memory layout (row-major):
 * Q, K, V, O: [batch, num_heads, seq_len, head_dim]
 *
 * Shared memory layout:
 * - Q_shared: [Br, d]
 * - K_shared: [Bc, d]
 * - V_shared: [Bc, d]
 * - S_shared: [Br, Bc] (attention scores)
 */
template<int BLOCK_SIZE, bool IS_CAUSAL>
__global__ void flash_attention_kernel(
    float* __restrict__ output,      // [B, H, N, d]
    const float* __restrict__ query, // [B, H, N, d]
    const float* __restrict__ key,   // [B, H_kv, N, d]
    const float* __restrict__ value, // [B, H_kv, N, d]
    int batch_size,
    int num_heads,
    int num_kv_heads,
    int seq_len,
    int head_dim,
    float scale
) {
    // Block handles one (batch, head) pair
    const int batch_idx = blockIdx.y;
    const int head_idx = blockIdx.x;
    const int kv_head_idx = head_idx / (num_heads / num_kv_heads);  // GQA mapping

    // Thread indices
    const int tid = threadIdx.x;
    const int num_threads = blockDim.x;

    // Shared memory
    extern __shared__ float shared_mem[];
    float* Q_shared = shared_mem;                           // [Br, d]
    float* K_shared = Q_shared + Br * d;                    // [Bc, d]
    float* V_shared = K_shared + Bc * d;                    // [Bc, d]
    float* S_shared = V_shared + Bc * d;                    // [Br, Bc]

    // Pointers to global memory for this batch/head
    const int q_offset = batch_idx * num_heads * seq_len * head_dim +
                         head_idx * seq_len * head_dim;
    const int kv_offset = batch_idx * num_kv_heads * seq_len * head_dim +
                          kv_head_idx * seq_len * head_dim;

    const float* Q_ptr = query + q_offset;
    const float* K_ptr = key + kv_offset;
    const float* V_ptr = value + kv_offset;
    float* O_ptr = output + q_offset;

    // Number of tiles
    const int num_q_tiles = (seq_len + Br - 1) / Br;
    const int num_kv_tiles = (seq_len + Bc - 1) / Bc;

    // Process each Q tile
    for (int q_tile = 0; q_tile < num_q_tiles; q_tile++) {
        const int q_start = q_tile * Br;

        // =====================================================================
        // Load Q tile into shared memory
        // =====================================================================
        for (int i = tid; i < Br * d; i += num_threads) {
            int row = i / d;
            int col = i % d;
            int global_row = q_start + row;

            if (global_row < seq_len) {
                Q_shared[row * d + col] = Q_ptr[global_row * head_dim + col] * scale;
            } else {
                Q_shared[row * d + col] = 0.0f;
            }
        }
        __syncthreads();

        // =====================================================================
        // Initialize output accumulators (per thread)
        // =====================================================================
        // Each thread handles Br/num_threads rows of Q
        const int rows_per_thread = (Br + num_threads - 1) / num_threads;

        float O_local[8][128];  // Local accumulator [rows_per_thread][d]
        float m_local[8];       // Running max
        float l_local[8];       // Running sum

        // Initialize
        for (int i = 0; i < rows_per_thread; i++) {
            m_local[i] = -FLT_MAX;
            l_local[i] = 0.0f;
            for (int j = 0; j < d; j++) {
                O_local[i][j] = 0.0f;
            }
        }

        // =====================================================================
        // Process each KV tile
        // =====================================================================
        for (int kv_tile = 0; kv_tile < num_kv_tiles; kv_tile++) {
            const int kv_start = kv_tile * Bc;

            // Causal mask: skip KV tiles entirely after Q positions
            if (IS_CAUSAL && kv_start > q_start + Br - 1) {
                break;
            }

            // -----------------------------------------------------------------
            // Load K tile
            // -----------------------------------------------------------------
            for (int i = tid; i < Bc * d; i += num_threads) {
                int row = i / d;
                int col = i % d;
                int global_row = kv_start + row;

                if (global_row < seq_len) {
                    K_shared[row * d + col] = K_ptr[global_row * head_dim + col];
                } else {
                    K_shared[row * d + col] = 0.0f;
                }
            }

            // -----------------------------------------------------------------
            // Load V tile
            // -----------------------------------------------------------------
            for (int i = tid; i < Bc * d; i += num_threads) {
                int row = i / d;
                int col = i % d;
                int global_row = kv_start + row;

                if (global_row < seq_len) {
                    V_shared[row * d + col] = V_ptr[global_row * head_dim + col];
                } else {
                    V_shared[row * d + col] = 0.0f;
                }
            }
            __syncthreads();

            // -----------------------------------------------------------------
            // Compute S = Q @ K^T for this tile
            // -----------------------------------------------------------------
            // Each thread computes a portion of the Br × Bc score matrix
            for (int i = tid; i < Br * Bc; i += num_threads) {
                int q_row = i / Bc;
                int k_col = i % Bc;

                float score = 0.0f;

                // Dot product
                #pragma unroll 8
                for (int j = 0; j < d; j++) {
                    score += Q_shared[q_row * d + j] * K_shared[k_col * d + j];
                }

                // Apply causal mask
                int global_q_pos = q_start + q_row;
                int global_k_pos = kv_start + k_col;

                if (IS_CAUSAL && global_k_pos > global_q_pos) {
                    score = -FLT_MAX;
                }

                S_shared[q_row * Bc + k_col] = score;
            }
            __syncthreads();

            // -----------------------------------------------------------------
            // Online softmax and accumulate output
            // -----------------------------------------------------------------
            // Each thread processes its assigned Q rows
            for (int local_row = 0; local_row < rows_per_thread; local_row++) {
                int q_row = tid * rows_per_thread + local_row;
                if (q_row >= Br) break;

                // Find max in this row
                float row_max = -FLT_MAX;
                for (int k = 0; k < Bc; k++) {
                    row_max = fmaxf(row_max, S_shared[q_row * Bc + k]);
                }

                // Update running max
                float m_new = fmaxf(m_local[local_row], row_max);

                // Compute exp(S - m_new) and sum
                float row_sum = 0.0f;
                float P_row[Bc];
                for (int k = 0; k < Bc; k++) {
                    P_row[k] = expf(S_shared[q_row * Bc + k] - m_new);
                    row_sum += P_row[k];
                }

                // Rescale previous output
                float scale_old = expf(m_local[local_row] - m_new);

                // Update running sum
                float l_new = l_local[local_row] * scale_old + row_sum;

                // Update output: O = O * scale + P @ V
                for (int j = 0; j < d; j++) {
                    O_local[local_row][j] *= scale_old;
                    for (int k = 0; k < Bc; k++) {
                        O_local[local_row][j] += P_row[k] * V_shared[k * d + j];
                    }
                }

                m_local[local_row] = m_new;
                l_local[local_row] = l_new;
            }
            __syncthreads();
        }

        // =====================================================================
        // Write output (normalize by l)
        // =====================================================================
        for (int local_row = 0; local_row < rows_per_thread; local_row++) {
            int q_row = tid * rows_per_thread + local_row;
            if (q_row >= Br) break;

            int global_row = q_start + q_row;
            if (global_row < seq_len) {
                float inv_l = 1.0f / l_local[local_row];
                for (int j = 0; j < d; j++) {
                    O_ptr[global_row * head_dim + j] = O_local[local_row][j] * inv_l;
                }
            }
        }
        __syncthreads();
    }
}

// =============================================================================
// Wrapper Functions
// =============================================================================

void flash_attention_forward(
    float* output,
    const float* query,
    const float* key,
    const float* value,
    int batch_size,
    int num_heads,
    int num_kv_heads,
    int seq_len,
    int head_dim,
    float scale,
    bool is_causal,
    cudaStream_t stream
) {
    // Calculate shared memory size
    size_t shared_mem_size = (Br * d + Bc * d + Bc * d + Br * Bc) * sizeof(float);

    // Grid: (num_heads, batch_size)
    // Block: 256 threads
    dim3 grid(num_heads, batch_size);
    dim3 block(256);

    if (is_causal) {
        flash_attention_kernel<256, true><<<grid, block, shared_mem_size, stream>>>(
            output, query, key, value,
            batch_size, num_heads, num_kv_heads, seq_len, head_dim, scale
        );
    } else {
        flash_attention_kernel<256, false><<<grid, block, shared_mem_size, stream>>>(
            output, query, key, value,
            batch_size, num_heads, num_kv_heads, seq_len, head_dim, scale
        );
    }

    CUDA_CHECK_LAST();
}

// =============================================================================
// Optimized Flash Attention with Better Parallelism
// =============================================================================

/**
 * flash_attention_v2_kernel - Improved parallelism
 *
 * Instead of one block per (batch, head), we use:
 * - One block per (batch, head, q_tile)
 * - Better GPU utilization for long sequences
 */
template<int BLOCK_SIZE, bool IS_CAUSAL>
__global__ void flash_attention_v2_kernel(
    float* __restrict__ output,
    float* __restrict__ output_lse,  // Log-sum-exp for later combination
    const float* __restrict__ query,
    const float* __restrict__ key,
    const float* __restrict__ value,
    int batch_size,
    int num_heads,
    int num_kv_heads,
    int seq_len,
    int head_dim,
    float scale,
    int num_q_tiles
) {
    // Block indices
    const int batch_idx = blockIdx.z;
    const int head_idx = blockIdx.y;
    const int q_tile_idx = blockIdx.x;
    const int kv_head_idx = head_idx / (num_heads / num_kv_heads);

    const int tid = threadIdx.x;

    // Shared memory
    extern __shared__ float shared_mem[];
    float* Q_shared = shared_mem;
    float* K_shared = Q_shared + Br * d;
    float* V_shared = K_shared + Bc * d;
    float* S_shared = V_shared + Bc * d;
    float* O_shared = S_shared + Br * Bc;
    float* m_shared = O_shared + Br * d;  // [Br]
    float* l_shared = m_shared + Br;       // [Br]

    // Global memory pointers
    const int q_offset = batch_idx * num_heads * seq_len * head_dim +
                         head_idx * seq_len * head_dim;
    const int kv_offset = batch_idx * num_kv_heads * seq_len * head_dim +
                          kv_head_idx * seq_len * head_dim;

    const float* Q_ptr = query + q_offset;
    const float* K_ptr = key + kv_offset;
    const float* V_ptr = value + kv_offset;
    float* O_ptr = output + q_offset;

    const int q_start = q_tile_idx * Br;
    const int num_kv_tiles = (seq_len + Bc - 1) / Bc;

    // Load Q tile (scaled)
    for (int i = tid; i < Br * d; i += BLOCK_SIZE) {
        int row = i / d;
        int col = i % d;
        int global_row = q_start + row;

        Q_shared[i] = (global_row < seq_len) ?
                      Q_ptr[global_row * head_dim + col] * scale : 0.0f;
    }

    // Initialize O, m, l
    for (int i = tid; i < Br * d; i += BLOCK_SIZE) {
        O_shared[i] = 0.0f;
    }
    for (int i = tid; i < Br; i += BLOCK_SIZE) {
        m_shared[i] = -FLT_MAX;
        l_shared[i] = 0.0f;
    }
    __syncthreads();

    // Process KV tiles
    for (int kv_tile = 0; kv_tile < num_kv_tiles; kv_tile++) {
        const int kv_start = kv_tile * Bc;

        if (IS_CAUSAL && kv_start > q_start + Br - 1) break;

        // Load K, V tiles
        for (int i = tid; i < Bc * d; i += BLOCK_SIZE) {
            int row = i / d;
            int global_row = kv_start + row;
            K_shared[i] = (global_row < seq_len) ? K_ptr[global_row * head_dim + i % d] : 0.0f;
            V_shared[i] = (global_row < seq_len) ? V_ptr[global_row * head_dim + i % d] : 0.0f;
        }
        __syncthreads();

        // Compute scores: S = Q @ K^T
        for (int i = tid; i < Br * Bc; i += BLOCK_SIZE) {
            int qr = i / Bc;
            int kc = i % Bc;

            float score = 0.0f;
            for (int j = 0; j < d; j++) {
                score += Q_shared[qr * d + j] * K_shared[kc * d + j];
            }

            // Causal mask
            if (IS_CAUSAL && (kv_start + kc) > (q_start + qr)) {
                score = -FLT_MAX;
            }

            S_shared[i] = score;
        }
        __syncthreads();

        // Row-wise max and softmax
        for (int row = tid; row < Br; row += BLOCK_SIZE) {
            float row_max = -FLT_MAX;
            for (int k = 0; k < Bc; k++) {
                row_max = fmaxf(row_max, S_shared[row * Bc + k]);
            }

            float m_new = fmaxf(m_shared[row], row_max);
            float scale_old = expf(m_shared[row] - m_new);

            float row_sum = 0.0f;
            for (int k = 0; k < Bc; k++) {
                float p = expf(S_shared[row * Bc + k] - m_new);
                S_shared[row * Bc + k] = p;  // Store P in S
                row_sum += p;
            }

            float l_new = l_shared[row] * scale_old + row_sum;

            // Update O
            for (int j = 0; j < d; j++) {
                float o = O_shared[row * d + j] * scale_old;
                for (int k = 0; k < Bc; k++) {
                    o += S_shared[row * Bc + k] * V_shared[k * d + j];
                }
                O_shared[row * d + j] = o;
            }

            m_shared[row] = m_new;
            l_shared[row] = l_new;
        }
        __syncthreads();
    }

    // Write output (normalized)
    for (int i = tid; i < Br; i += BLOCK_SIZE) {
        int global_row = q_start + i;
        if (global_row < seq_len) {
            float inv_l = 1.0f / l_shared[i];
            for (int j = 0; j < d; j++) {
                O_ptr[global_row * head_dim + j] = O_shared[i * d + j] * inv_l;
            }
        }
    }
}

} // namespace mini_vllm
```

---

## GQA Support

**Grouped Query Attention** shares KV heads across multiple Q heads:

```
                    GQA in Flash Attention

Standard MHA (32 Q heads, 32 KV heads):
Q Head 0  ────────────────  KV Head 0
Q Head 1  ────────────────  KV Head 1
   ...
Q Head 31 ────────────────  KV Head 31


GQA (32 Q heads, 8 KV heads):
Q Head 0  ──┐
Q Head 1  ──┼──────────────  KV Head 0
Q Head 2  ──┤
Q Head 3  ──┘
Q Head 4  ──┐
Q Head 5  ──┼──────────────  KV Head 1
Q Head 6  ──┤
Q Head 7  ──┘
   ...

Mapping: kv_head_idx = q_head_idx / (num_q_heads / num_kv_heads)
         kv_head_idx = q_head_idx / 4  (for ratio=4)
```

This is already handled in our kernel:

```c++
const int kv_head_idx = head_idx / (num_heads / num_kv_heads);
```

---

## Testing and Verification

Create file: `mini_vllm/tests/cpp/test_flash_attention.cu`

```c++
// =============================================================================
// test_flash_attention.cu - Flash Attention Unit Tests
// =============================================================================

#include "flash_attention.cuh"
#include <cuda_runtime.h>
#include <cstdio>
#include <cmath>
#include <vector>
#include <random>

using namespace mini_vllm;

// CPU reference (standard attention)
void attention_cpu(
    float* output,
    const float* query,
    const float* key,
    const float* value,
    int batch_size,
    int num_heads,
    int seq_len,
    int head_dim,
    float scale,
    bool is_causal
) {
    for (int b = 0; b < batch_size; b++) {
        for (int h = 0; h < num_heads; h++) {
            const float* Q = query + b * num_heads * seq_len * head_dim +
                             h * seq_len * head_dim;
            const float* K = key + b * num_heads * seq_len * head_dim +
                             h * seq_len * head_dim;
            const float* V = value + b * num_heads * seq_len * head_dim +
                             h * seq_len * head_dim;
            float* O = output + b * num_heads * seq_len * head_dim +
                       h * seq_len * head_dim;

            // For each query position
            for (int i = 0; i < seq_len; i++) {
                // Compute scores and max
                std::vector<float> scores(seq_len);
                float max_score = -FLT_MAX;

                for (int j = 0; j < seq_len; j++) {
                    float score = 0.0f;
                    for (int d = 0; d < head_dim; d++) {
                        score += Q[i * head_dim + d] * K[j * head_dim + d];
                    }
                    score *= scale;

                    // Causal mask
                    if (is_causal && j > i) {
                        score = -FLT_MAX;
                    }

                    scores[j] = score;
                    max_score = fmaxf(max_score, score);
                }

                // Softmax
                float sum_exp = 0.0f;
                for (int j = 0; j < seq_len; j++) {
                    scores[j] = expf(scores[j] - max_score);
                    sum_exp += scores[j];
                }
                for (int j = 0; j < seq_len; j++) {
                    scores[j] /= sum_exp;
                }

                // Apply attention
                for (int d = 0; d < head_dim; d++) {
                    float out = 0.0f;
                    for (int j = 0; j < seq_len; j++) {
                        out += scores[j] * V[j * head_dim + d];
                    }
                    O[i * head_dim + d] = out;
                }
            }
        }
    }
}

float max_abs_error(const float* a, const float* b, int n) {
    float max_err = 0.0f;
    for (int i = 0; i < n; i++) {
        float err = fabsf(a[i] - b[i]);
        if (err > max_err) max_err = err;
    }
    return max_err;
}

int main() {
    printf("=== Flash Attention Tests ===\n\n");

    // Configuration
    const int batch_size = 2;
    const int num_heads = 32;
    const int num_kv_heads = 8;
    const int seq_len = 256;  // Small for testing
    const int head_dim = 128;
    const float scale = 1.0f / sqrtf(head_dim);

    const int total_size = batch_size * num_heads * seq_len * head_dim;
    const int kv_size = batch_size * num_kv_heads * seq_len * head_dim;

    // Allocate host memory
    std::vector<float> h_q(total_size);
    std::vector<float> h_k(kv_size);
    std::vector<float> h_v(kv_size);
    std::vector<float> h_o_cpu(total_size);
    std::vector<float> h_o_gpu(total_size);

    // Initialize
    std::mt19937 gen(42);
    std::normal_distribution<float> dist(0.0f, 0.5f);

    for (auto& v : h_q) v = dist(gen);
    for (auto& v : h_k) v = dist(gen);
    for (auto& v : h_v) v = dist(gen);

    // For testing, use MHA (same Q and KV heads) for CPU comparison
    // Expand K and V to match Q heads
    std::vector<float> h_k_expanded(total_size);
    std::vector<float> h_v_expanded(total_size);

    for (int b = 0; b < batch_size; b++) {
        for (int h = 0; h < num_heads; h++) {
            int kv_h = h / (num_heads / num_kv_heads);
            for (int s = 0; s < seq_len; s++) {
                for (int d = 0; d < head_dim; d++) {
                    int q_idx = b * num_heads * seq_len * head_dim +
                                h * seq_len * head_dim + s * head_dim + d;
                    int kv_idx = b * num_kv_heads * seq_len * head_dim +
                                 kv_h * seq_len * head_dim + s * head_dim + d;
                    h_k_expanded[q_idx] = h_k[kv_idx];
                    h_v_expanded[q_idx] = h_v[kv_idx];
                }
            }
        }
    }

    // Allocate device memory
    float *d_q, *d_k, *d_v, *d_o;
    cudaMalloc(&d_q, total_size * sizeof(float));
    cudaMalloc(&d_k, kv_size * sizeof(float));
    cudaMalloc(&d_v, kv_size * sizeof(float));
    cudaMalloc(&d_o, total_size * sizeof(float));

    cudaMemcpy(d_q, h_q.data(), total_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_k, h_k.data(), kv_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_v, h_v.data(), kv_size * sizeof(float), cudaMemcpyHostToDevice);

    // =======================================================================
    // Test 1: Correctness (causal attention)
    // =======================================================================
    printf("Test 1: Correctness (causal)\n");

    // CPU reference with expanded KV
    attention_cpu(h_o_cpu.data(), h_q.data(),
                  h_k_expanded.data(), h_v_expanded.data(),
                  batch_size, num_heads, seq_len, head_dim, scale, true);

    // GPU
    flash_attention_forward(d_o, d_q, d_k, d_v,
                            batch_size, num_heads, num_kv_heads,
                            seq_len, head_dim, scale, true);
    cudaDeviceSynchronize();

    cudaMemcpy(h_o_gpu.data(), d_o, total_size * sizeof(float),
               cudaMemcpyDeviceToHost);

    float err = max_abs_error(h_o_cpu.data(), h_o_gpu.data(), total_size);
    printf("  Max absolute error: %.2e\n", err);
    printf("  %s\n", err < 1e-3f ? "[PASS]" : "[FAIL]");

    // =======================================================================
    // Test 2: Causal mask verification
    // =======================================================================
    printf("\nTest 2: Causal mask verification\n");

    // For causal attention, output[i] should not depend on input[j] for j > i
    // We verify by checking that changing future tokens doesn't affect past outputs

    // Modify last token's K
    std::vector<float> h_k_modified = h_k;
    for (int b = 0; b < batch_size; b++) {
        for (int h = 0; h < num_kv_heads; h++) {
            int base = b * num_kv_heads * seq_len * head_dim +
                       h * seq_len * head_dim + (seq_len - 1) * head_dim;
            for (int d = 0; d < head_dim; d++) {
                h_k_modified[base + d] = 999.0f;  // Extreme change
            }
        }
    }

    float* d_k_mod;
    cudaMalloc(&d_k_mod, kv_size * sizeof(float));
    cudaMemcpy(d_k_mod, h_k_modified.data(), kv_size * sizeof(float),
               cudaMemcpyHostToDevice);

    std::vector<float> h_o_modified(total_size);
    flash_attention_forward(d_o, d_q, d_k_mod, d_v,
                            batch_size, num_heads, num_kv_heads,
                            seq_len, head_dim, scale, true);
    cudaDeviceSynchronize();

    cudaMemcpy(h_o_modified.data(), d_o, total_size * sizeof(float),
               cudaMemcpyDeviceToHost);

    // Check that outputs before last position are unchanged
    float causal_err = 0.0f;
    for (int b = 0; b < batch_size; b++) {
        for (int h = 0; h < num_heads; h++) {
            for (int s = 0; s < seq_len - 1; s++) {  // All but last
                for (int d = 0; d < head_dim; d++) {
                    int idx = b * num_heads * seq_len * head_dim +
                              h * seq_len * head_dim + s * head_dim + d;
                    causal_err = fmaxf(causal_err,
                                       fabsf(h_o_gpu[idx] - h_o_modified[idx]));
                }
            }
        }
    }

    printf("  Max error in past positions: %.2e\n", causal_err);
    printf("  %s\n", causal_err < 1e-5f ? "[PASS] Causal mask working" : "[FAIL]");

    cudaFree(d_k_mod);

    // =======================================================================
    // Test 3: Performance
    // =======================================================================
    printf("\nTest 3: Performance\n");

    // Warmup
    for (int i = 0; i < 10; i++) {
        flash_attention_forward(d_o, d_q, d_k, d_v,
                                batch_size, num_heads, num_kv_heads,
                                seq_len, head_dim, scale, true);
    }
    cudaDeviceSynchronize();

    // Benchmark
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    const int iterations = 100;
    cudaEventRecord(start);
    for (int i = 0; i < iterations; i++) {
        flash_attention_forward(d_o, d_q, d_k, d_v,
                                batch_size, num_heads, num_kv_heads,
                                seq_len, head_dim, scale, true);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    float avg_ms = ms / iterations;

    // Compute FLOPS
    // For each query position: 2 * seq_len * head_dim (Q@K) + 2 * seq_len * head_dim (P@V)
    float flops = batch_size * num_heads * seq_len *
                  (4.0f * seq_len * head_dim) * iterations;
    float tflops = flops / (ms * 1e9f);

    printf("  Batch=%d, Heads=%d, SeqLen=%d, HeadDim=%d\n",
           batch_size, num_heads, seq_len, head_dim);
    printf("  Time: %.3f ms\n", avg_ms);
    printf("  Throughput: %.2f TFLOP/s\n", tflops);

    // Cleanup
    cudaFree(d_q);
    cudaFree(d_k);
    cudaFree(d_v);
    cudaFree(d_o);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    printf("\n=== Tests Complete ===\n");
    return 0;
}
```

### Compilation

```bash
cd ~/work/mini_vllm
nvcc -O3 -std=c++17 \
    -I csrc/include \
    -o test_flash_attention \
    tests/cpp/test_flash_attention.cu \
    csrc/attention/flash_attention.cu

./test_flash_attention
```

---

## Summary

You've implemented Flash Attention with:

| Feature               | Implementation        |
| --------------------- | --------------------- |
| **Tiled computation** | Br×Bc tiles in SRAM   |
| **Online softmax**    | Running max and sum   |
| **GQA support**       | KV head sharing       |
| **Causal masking**    | Skip future positions |

### Key Optimizations Missing (for production)

1. **Tensor Cores** - Use WMMA/MMA for matrix multiply
2. **Software pipelining** - Overlap memory loads with compute
3. **FP16 with FP32 accumulator** - Higher bandwidth
4. **Split-KV parallelism** - More blocks for long sequences

---

## What's Next

Next, we'll implement **FlashInfer for Decode**, which is optimized for single-token attention during generation.

Continue to: [07_flash_infer_decode.md](./07_flash_infer_decode.md)
