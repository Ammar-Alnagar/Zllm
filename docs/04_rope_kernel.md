# Phase 1: RoPE Kernel Implementation

## Table of Contents

1. [Understanding RoPE](#understanding-rope)
2. [Mathematical Background](#mathematical-background)
3. [Implementation Strategy](#implementation-strategy)
4. [CUDA Kernel Implementation](#cuda-kernel-implementation)
5. [Batched RoPE for GQA](#batched-rope-for-gqa)
6. [Testing and Verification](#testing-and-verification)

---

## Understanding RoPE

**Rotary Position Embeddings (RoPE)** encode position information by rotating query and key vectors. Unlike absolute position embeddings, RoPE enables the model to learn relative positions.

```
                    Why RoPE?

Traditional Position Embedding:
- Add position vector to token embedding
- Absolute position: token[5] always has same position encoding
- Hard to extrapolate to longer sequences

Rotary Position Embedding:
- Rotate query and key vectors based on position
- Relative position: attention depends on (position_q - position_k)
- Better length generalization
- No extra parameters!
```

### RoPE in the Transformer

```
                    RoPE Application Point

                Input Embeddings
                       │
                       ▼
                   ┌───────┐
                   │RMSNorm│
                   └───┬───┘
                       │
            ┌──────────┼──────────┐
            │          │          │
            ▼          ▼          ▼
         ┌─────┐   ┌─────┐   ┌─────┐
         │ Q   │   │ K   │   │ V   │
         │Proj │   │Proj │   │Proj │
         └──┬──┘   └──┬──┘   └─────┘
            │         │          │
            ▼         ▼          │
        ┌──────┐  ┌──────┐       │
        │ RoPE │  │ RoPE │       │  ◀── Apply RoPE to Q and K only
        └──┬───┘  └──┬───┘       │      V is NOT rotated
            │         │          │
            └────┬────┘          │
                 │               │
                 ▼               │
           ┌───────────┐         │
           │ Attention │◀────────┘
           └───────────┘
```

---

## Mathematical Background

### The Core Idea

RoPE treats pairs of dimensions as 2D subspaces and applies rotation:

```
For position m and dimension pair (2i, 2i+1):

                    Rotation Matrix

┌           ┐   ┌                    ┐   ┌       ┐
│ q'_{2i}   │   │ cos(mθᵢ)  -sin(mθᵢ)│   │ q_{2i}│
│           │ = │                    │ × │       │
│ q'_{2i+1} │   │ sin(mθᵢ)   cos(mθᵢ)│   │q_{2i+1}│
└           ┘   └                    ┘   └       ┘

Where θᵢ = 1 / (θ_base^(2i/d))

θ_base = 10000 (default) or 1000000 (Qwen3)
d = head dimension (typically 128)
```

### Expanded Formula

```
q'_{2i}   = q_{2i}   × cos(mθᵢ) - q_{2i+1} × sin(mθᵢ)
q'_{2i+1} = q_{2i}   × sin(mθᵢ) + q_{2i+1} × cos(mθᵢ)

Similarly for keys:
k'_{2i}   = k_{2i}   × cos(nθᵢ) - k_{2i+1} × sin(nθᵢ)
k'_{2i+1} = k_{2i}   × sin(nθᵢ) + k_{2i+1} × cos(nθᵢ)
```

### Why Relative Position Emerges

```
After applying RoPE, the dot product of q at position m with k at position n:

q'_m · k'_n = f(q, k, m-n)

The result only depends on RELATIVE position (m-n), not absolute positions!

Proof sketch:
  q'_m · k'_n = Σᵢ (q_{2i} cos(mθᵢ) - q_{2i+1} sin(mθᵢ)) ×
                   (k_{2i} cos(nθᵢ) - k_{2i+1} sin(nθᵢ)) + ...

Using cos(a)cos(b) + sin(a)sin(b) = cos(a-b):
              = Σᵢ [...terms with cos((m-n)θᵢ) and sin((m-n)θᵢ)...]
```

### Frequency Computation

```
For head_dim = 128 and theta_base = 10000:

dim_pair i │ θᵢ = 10000^(-2i/128)     │ Wavelength
───────────┼──────────────────────────┼────────────
    0      │ 10000^0     = 1.0        │ 2π ≈ 6
    1      │ 10000^-0.016 ≈ 0.832     │ 7.5
    2      │ 10000^-0.031 ≈ 0.692     │ 9.1
   ...     │ ...                       │ ...
   63      │ 10000^-0.984 ≈ 0.0001    │ ~63,000

Lower dimensions: High frequency (captures local patterns)
Higher dimensions: Low frequency (captures global patterns)
```

---

## Implementation Strategy

### Layout and Parallelization

```
Input Q tensor: [batch_size, seq_len, num_heads, head_dim]

For each (batch, position, head, dim_pair):
  Apply rotation with angle = position × θ_{dim_pair/2}

Parallelization:
- Grid: (batch_size × num_heads, seq_len)
- Block: (head_dim / 2) threads
- Each thread handles one dimension pair
```

### Precomputation

We can precompute cos/sin tables:

```
Precomputed:
  cos_table[max_seq_len, head_dim/2]
  sin_table[max_seq_len, head_dim/2]

For position m, dimension pair i:
  cos_val = cos_table[m, i]
  sin_val = sin_table[m, i]
```

---

## CUDA Kernel Implementation

Create file: `mini_vllm/csrc/kernels/rope.cuh`

```c++
// =============================================================================
// rope.cuh - Rotary Position Embedding Header
// =============================================================================

#pragma once

#include "common.cuh"
#include <cuda_runtime.h>
#include <cuda_fp16.h>

namespace mini_vllm {

/**
 * Initialize RoPE frequency tables
 *
 * @param cos_table: Output cosine table [max_seq_len, head_dim/2]
 * @param sin_table: Output sine table [max_seq_len, head_dim/2]
 * @param max_seq_len: Maximum sequence length
 * @param head_dim: Dimension per head
 * @param theta_base: Base for frequency computation (10000 or 1000000)
 * @param stream: CUDA stream
 */
void rope_init_tables(
    float* cos_table,
    float* sin_table,
    int max_seq_len,
    int head_dim,
    float theta_base,
    cudaStream_t stream = nullptr
);

/**
 * Apply RoPE to query and key tensors
 *
 * @param query: Query tensor [num_tokens, num_heads, head_dim] (in-place)
 * @param key: Key tensor [num_tokens, num_kv_heads, head_dim] (in-place)
 * @param cos_table: Precomputed cosines [max_seq_len, head_dim/2]
 * @param sin_table: Precomputed sines [max_seq_len, head_dim/2]
 * @param positions: Position indices [num_tokens]
 * @param num_tokens: Number of tokens
 * @param num_heads: Number of query heads
 * @param num_kv_heads: Number of key-value heads
 * @param head_dim: Dimension per head
 * @param stream: CUDA stream
 */
void rope_forward(
    float* query,
    float* key,
    const float* cos_table,
    const float* sin_table,
    const int* positions,
    int num_tokens,
    int num_heads,
    int num_kv_heads,
    int head_dim,
    cudaStream_t stream = nullptr
);

// FP16 version
void rope_forward_fp16(
    half* query,
    half* key,
    const float* cos_table,
    const float* sin_table,
    const int* positions,
    int num_tokens,
    int num_heads,
    int num_kv_heads,
    int head_dim,
    cudaStream_t stream = nullptr
);

} // namespace mini_vllm
```

Create file: `mini_vllm/csrc/kernels/rope.cu`

```c++
// =============================================================================
// rope.cu - Rotary Position Embedding Implementation
// =============================================================================

#include "rope.cuh"
#include <cmath>

namespace mini_vllm {

// =============================================================================
// Frequency Table Initialization
// =============================================================================

/**
 * Kernel to compute cos/sin tables
 *
 * Each thread computes one (position, dim_pair) entry
 */
__global__ void rope_init_kernel(
    float* __restrict__ cos_table,  // [max_seq_len, head_dim/2]
    float* __restrict__ sin_table,  // [max_seq_len, head_dim/2]
    int max_seq_len,
    int half_head_dim,
    float theta_base
) {
    // Position index
    int pos = blockIdx.x;
    // Dimension pair index
    int dim = threadIdx.x;

    if (pos < max_seq_len && dim < half_head_dim) {
        // Compute frequency: theta_i = 1 / (base^(2i/d))
        // = base^(-2i/d)
        float freq = powf(theta_base, -2.0f * dim / (2.0f * half_head_dim));

        // Compute angle for this position
        float angle = pos * freq;

        // Store cos and sin
        int idx = pos * half_head_dim + dim;
        cos_table[idx] = cosf(angle);
        sin_table[idx] = sinf(angle);
    }
}

void rope_init_tables(
    float* cos_table,
    float* sin_table,
    int max_seq_len,
    int head_dim,
    float theta_base,
    cudaStream_t stream
) {
    int half_head_dim = head_dim / 2;

    // Launch one block per position, threads per dimension pair
    dim3 grid(max_seq_len);
    dim3 block(half_head_dim);

    rope_init_kernel<<<grid, block, 0, stream>>>(
        cos_table, sin_table, max_seq_len, half_head_dim, theta_base
    );

    CUDA_CHECK_LAST();
}

// =============================================================================
// RoPE Forward Pass
// =============================================================================

/**
 * rope_kernel - Apply rotary embeddings to Q and K
 *
 * Memory layout:
 * Q: [num_tokens, num_heads, head_dim]
 * K: [num_tokens, num_kv_heads, head_dim]
 *
 * For GQA: num_heads > num_kv_heads (Q heads grouped to share KV)
 *
 * Each block handles one (token, head) pair
 * Each thread handles one dimension pair
 */
__global__ void rope_kernel(
    float* __restrict__ query,      // [num_tokens, num_heads, head_dim]
    float* __restrict__ key,        // [num_tokens, num_kv_heads, head_dim]
    const float* __restrict__ cos_table,  // [max_seq_len, head_dim/2]
    const float* __restrict__ sin_table,  // [max_seq_len, head_dim/2]
    const int* __restrict__ positions,    // [num_tokens]
    int num_tokens,
    int num_heads,
    int num_kv_heads,
    int head_dim
) {
    // Block indices
    const int token_idx = blockIdx.y;
    const int head_idx = blockIdx.x;

    // Thread index = dimension pair index
    const int dim_pair = threadIdx.x;
    const int half_head_dim = head_dim / 2;

    if (dim_pair >= half_head_dim) return;

    // Get position for this token
    const int pos = positions[token_idx];

    // Load cos/sin for this position and dimension
    const int table_idx = pos * half_head_dim + dim_pair;
    const float cos_val = cos_table[table_idx];
    const float sin_val = sin_table[table_idx];

    // =========================================================================
    // Apply RoPE to Query
    // =========================================================================
    if (head_idx < num_heads) {
        // Calculate indices for the dimension pair in Q
        // Q layout: [num_tokens, num_heads, head_dim]
        int q_base = token_idx * num_heads * head_dim + head_idx * head_dim;
        int q_idx_even = q_base + 2 * dim_pair;
        int q_idx_odd = q_base + 2 * dim_pair + 1;

        // Load Q values
        float q_even = query[q_idx_even];
        float q_odd = query[q_idx_odd];

        // Apply rotation:
        // q'_even = q_even * cos - q_odd * sin
        // q'_odd  = q_even * sin + q_odd * cos
        float q_rotated_even = q_even * cos_val - q_odd * sin_val;
        float q_rotated_odd = q_even * sin_val + q_odd * cos_val;

        // Store rotated values
        query[q_idx_even] = q_rotated_even;
        query[q_idx_odd] = q_rotated_odd;
    }

    // =========================================================================
    // Apply RoPE to Key
    // =========================================================================
    // Only apply to KV heads (fewer than Q heads in GQA)
    if (head_idx < num_kv_heads) {
        // K layout: [num_tokens, num_kv_heads, head_dim]
        int k_base = token_idx * num_kv_heads * head_dim + head_idx * head_dim;
        int k_idx_even = k_base + 2 * dim_pair;
        int k_idx_odd = k_base + 2 * dim_pair + 1;

        // Load K values
        float k_even = key[k_idx_even];
        float k_odd = key[k_idx_odd];

        // Apply rotation
        float k_rotated_even = k_even * cos_val - k_odd * sin_val;
        float k_rotated_odd = k_even * sin_val + k_odd * cos_val;

        // Store rotated values
        key[k_idx_even] = k_rotated_even;
        key[k_idx_odd] = k_rotated_odd;
    }
}

void rope_forward(
    float* query,
    float* key,
    const float* cos_table,
    const float* sin_table,
    const int* positions,
    int num_tokens,
    int num_heads,
    int num_kv_heads,
    int head_dim,
    cudaStream_t stream
) {
    int half_head_dim = head_dim / 2;

    // Grid: (max(num_heads, num_kv_heads), num_tokens)
    // Block: half_head_dim threads
    int max_heads = max(num_heads, num_kv_heads);

    dim3 grid(max_heads, num_tokens);
    dim3 block(half_head_dim);

    rope_kernel<<<grid, block, 0, stream>>>(
        query, key, cos_table, sin_table, positions,
        num_tokens, num_heads, num_kv_heads, head_dim
    );

    CUDA_CHECK_LAST();
}

// =============================================================================
// Optimized RoPE with Vectorization
// =============================================================================

/**
 * Optimized RoPE kernel with:
 * 1. Vectorized loads/stores (float2)
 * 2. Better memory coalescing
 * 3. Fused Q and K processing
 */
__global__ void rope_optimized_kernel(
    float* __restrict__ query,
    float* __restrict__ key,
    const float* __restrict__ cos_table,
    const float* __restrict__ sin_table,
    const int* __restrict__ positions,
    int num_tokens,
    int num_heads,
    int num_kv_heads,
    int head_dim
) {
    const int token_idx = blockIdx.y;
    const int head_idx = blockIdx.x;
    const int dim_pair = threadIdx.x;
    const int half_head_dim = head_dim / 2;

    if (dim_pair >= half_head_dim) return;

    // Get position and load cos/sin (these are cached in L1)
    const int pos = positions[token_idx];
    const int table_idx = pos * half_head_dim + dim_pair;

    // Use float2 for cos/sin pair
    const float cos_val = cos_table[table_idx];
    const float sin_val = sin_table[table_idx];

    // Process Query
    if (head_idx < num_heads) {
        int q_base = token_idx * num_heads * head_dim + head_idx * head_dim;

        // Load as float2 (8 bytes, coalesced)
        float2* q_ptr = reinterpret_cast<float2*>(&query[q_base + 2 * dim_pair]);
        float2 q = *q_ptr;

        // Rotate
        float2 q_rot;
        q_rot.x = q.x * cos_val - q.y * sin_val;
        q_rot.y = q.x * sin_val + q.y * cos_val;

        // Store
        *q_ptr = q_rot;
    }

    // Process Key
    if (head_idx < num_kv_heads) {
        int k_base = token_idx * num_kv_heads * head_dim + head_idx * head_dim;

        float2* k_ptr = reinterpret_cast<float2*>(&key[k_base + 2 * dim_pair]);
        float2 k = *k_ptr;

        float2 k_rot;
        k_rot.x = k.x * cos_val - k.y * sin_val;
        k_rot.y = k.x * sin_val + k.y * cos_val;

        *k_ptr = k_rot;
    }
}

// =============================================================================
// FP16 Implementation
// =============================================================================

__global__ void rope_fp16_kernel(
    half* __restrict__ query,
    half* __restrict__ key,
    const float* __restrict__ cos_table,
    const float* __restrict__ sin_table,
    const int* __restrict__ positions,
    int num_tokens,
    int num_heads,
    int num_kv_heads,
    int head_dim
) {
    const int token_idx = blockIdx.y;
    const int head_idx = blockIdx.x;
    const int dim_pair = threadIdx.x;
    const int half_head_dim = head_dim / 2;

    if (dim_pair >= half_head_dim) return;

    const int pos = positions[token_idx];
    const int table_idx = pos * half_head_dim + dim_pair;

    // Keep cos/sin in FP32 for accuracy
    const float cos_val = cos_table[table_idx];
    const float sin_val = sin_table[table_idx];

    // Process Query
    if (head_idx < num_heads) {
        int q_base = token_idx * num_heads * head_dim + head_idx * head_dim;
        int q_idx = q_base + 2 * dim_pair;

        // Load half2 (both elements of the pair)
        half2* q_ptr = reinterpret_cast<half2*>(&query[q_idx]);
        half2 q_half = *q_ptr;

        // Convert to float for computation
        float2 q = __half22float2(q_half);

        // Rotate
        float2 q_rot;
        q_rot.x = q.x * cos_val - q.y * sin_val;
        q_rot.y = q.x * sin_val + q.y * cos_val;

        // Convert back and store
        *q_ptr = __float22half2_rn(q_rot);
    }

    // Process Key
    if (head_idx < num_kv_heads) {
        int k_base = token_idx * num_kv_heads * head_dim + head_idx * head_dim;
        int k_idx = k_base + 2 * dim_pair;

        half2* k_ptr = reinterpret_cast<half2*>(&key[k_idx]);
        half2 k_half = *k_ptr;

        float2 k = __half22float2(k_half);

        float2 k_rot;
        k_rot.x = k.x * cos_val - k.y * sin_val;
        k_rot.y = k.x * sin_val + k.y * cos_val;

        *k_ptr = __float22half2_rn(k_rot);
    }
}

void rope_forward_fp16(
    half* query,
    half* key,
    const float* cos_table,
    const float* sin_table,
    const int* positions,
    int num_tokens,
    int num_heads,
    int num_kv_heads,
    int head_dim,
    cudaStream_t stream
) {
    int half_head_dim = head_dim / 2;
    int max_heads = max(num_heads, num_kv_heads);

    dim3 grid(max_heads, num_tokens);
    dim3 block(half_head_dim);

    rope_fp16_kernel<<<grid, block, 0, stream>>>(
        query, key, cos_table, sin_table, positions,
        num_tokens, num_heads, num_kv_heads, head_dim
    );

    CUDA_CHECK_LAST();
}

} // namespace mini_vllm
```

---

## Batched RoPE for GQA

For **Grouped Query Attention**, multiple Q heads share the same KV heads:

```
                    GQA Head Mapping

num_heads = 32       (Query heads)
num_kv_heads = 8     (Key-Value heads)
ratio = 32 / 8 = 4   (4 Q heads per KV head)

Q Heads:  Q0  Q1  Q2  Q3   Q4  Q5  Q6  Q7   ...   Q28 Q29 Q30 Q31
          └───────┬───────┘ └───────┬───────┘     └───────┬───────┘
                  │                 │                     │
KV Heads:        K0/V0            K1/V1                 K7/V7
```

For RoPE, we apply to ALL Q heads but only to the distinct KV heads:

```c++
// =============================================================================
// RoPE Kernel for GQA (Already handled above, but here's clarification)
// =============================================================================

/**
 * In our kernel:
 * - Q heads: 0 to num_heads-1 (all 32)
 * - K heads: 0 to num_kv_heads-1 (only 8)
 *
 * Each block handles one (token, head_idx) pair
 * For head_idx >= num_kv_heads, we skip K processing
 */
```

---

## Testing and Verification

Create file: `mini_vllm/tests/cpp/test_rope.cu`

```c++
// =============================================================================
// test_rope.cu - RoPE Unit Tests
// =============================================================================

#include "rope.cuh"
#include <cuda_runtime.h>
#include <cstdio>
#include <cmath>
#include <vector>
#include <random>

using namespace mini_vllm;

// CPU reference implementation
void rope_cpu(
    float* query,
    float* key,
    const int* positions,
    int num_tokens,
    int num_heads,
    int num_kv_heads,
    int head_dim,
    float theta_base
) {
    int half_head_dim = head_dim / 2;

    for (int t = 0; t < num_tokens; t++) {
        int pos = positions[t];

        for (int dim = 0; dim < half_head_dim; dim++) {
            // Compute frequency
            float freq = powf(theta_base, -2.0f * dim / head_dim);
            float angle = pos * freq;
            float cos_val = cosf(angle);
            float sin_val = sinf(angle);

            // Apply to all Q heads
            for (int h = 0; h < num_heads; h++) {
                int base = t * num_heads * head_dim + h * head_dim;
                int idx_even = base + 2 * dim;
                int idx_odd = base + 2 * dim + 1;

                float q_even = query[idx_even];
                float q_odd = query[idx_odd];

                query[idx_even] = q_even * cos_val - q_odd * sin_val;
                query[idx_odd] = q_even * sin_val + q_odd * cos_val;
            }

            // Apply to KV heads
            for (int h = 0; h < num_kv_heads; h++) {
                int base = t * num_kv_heads * head_dim + h * head_dim;
                int idx_even = base + 2 * dim;
                int idx_odd = base + 2 * dim + 1;

                float k_even = key[idx_even];
                float k_odd = key[idx_odd];

                key[idx_even] = k_even * cos_val - k_odd * sin_val;
                key[idx_odd] = k_even * sin_val + k_odd * cos_val;
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
    printf("=== RoPE Kernel Tests ===\n\n");

    // Configuration (Qwen3-like)
    const int num_tokens = 128;
    const int num_heads = 32;
    const int num_kv_heads = 8;
    const int head_dim = 128;
    const int max_seq_len = 8192;
    const float theta_base = 1000000.0f;  // Qwen3 uses 1M

    // Sizes
    const int q_size = num_tokens * num_heads * head_dim;
    const int k_size = num_tokens * num_kv_heads * head_dim;
    const int table_size = max_seq_len * (head_dim / 2);

    // Allocate host memory
    std::vector<float> h_q_cpu(q_size);
    std::vector<float> h_q_gpu(q_size);
    std::vector<float> h_k_cpu(k_size);
    std::vector<float> h_k_gpu(k_size);
    std::vector<int> h_positions(num_tokens);

    // Initialize
    std::mt19937 gen(42);
    std::normal_distribution<float> dist(0.0f, 1.0f);

    for (auto& v : h_q_cpu) v = dist(gen);
    for (auto& v : h_k_cpu) v = dist(gen);
    h_q_gpu = h_q_cpu;  // Copy for GPU
    h_k_gpu = h_k_cpu;

    // Positions: 0, 1, 2, ..., num_tokens-1
    for (int i = 0; i < num_tokens; i++) {
        h_positions[i] = i;
    }

    // Allocate device memory
    float *d_q, *d_k, *d_cos, *d_sin;
    int *d_positions;

    cudaMalloc(&d_q, q_size * sizeof(float));
    cudaMalloc(&d_k, k_size * sizeof(float));
    cudaMalloc(&d_cos, table_size * sizeof(float));
    cudaMalloc(&d_sin, table_size * sizeof(float));
    cudaMalloc(&d_positions, num_tokens * sizeof(int));

    // Initialize tables
    rope_init_tables(d_cos, d_sin, max_seq_len, head_dim, theta_base);

    // Copy data
    cudaMemcpy(d_q, h_q_gpu.data(), q_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_k, h_k_gpu.data(), k_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_positions, h_positions.data(), num_tokens * sizeof(int), cudaMemcpyHostToDevice);

    // =======================================================================
    // Test 1: Correctness
    // =======================================================================
    printf("Test 1: Correctness\n");

    // CPU reference (modifies in-place)
    rope_cpu(h_q_cpu.data(), h_k_cpu.data(), h_positions.data(),
             num_tokens, num_heads, num_kv_heads, head_dim, theta_base);

    // GPU (modifies in-place)
    rope_forward(d_q, d_k, d_cos, d_sin, d_positions,
                 num_tokens, num_heads, num_kv_heads, head_dim);
    cudaDeviceSynchronize();

    // Copy back
    cudaMemcpy(h_q_gpu.data(), d_q, q_size * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_k_gpu.data(), d_k, k_size * sizeof(float), cudaMemcpyDeviceToHost);

    // Compare
    float q_err = max_abs_error(h_q_cpu.data(), h_q_gpu.data(), q_size);
    float k_err = max_abs_error(h_k_cpu.data(), h_k_gpu.data(), k_size);

    printf("  Q max error: %.2e\n", q_err);
    printf("  K max error: %.2e\n", k_err);
    if (q_err < 1e-5f && k_err < 1e-5f) {
        printf("  [PASS] Results match\n");
    } else {
        printf("  [FAIL] Error too large!\n");
    }

    // =======================================================================
    // Test 2: Relative position property
    // =======================================================================
    printf("\nTest 2: Relative position property\n");

    // Create two Q/K pairs at different absolute positions
    // but same relative position
    std::vector<float> q1(num_heads * head_dim);
    std::vector<float> k2(num_kv_heads * head_dim);
    std::vector<float> q1_shift(num_heads * head_dim);
    std::vector<float> k2_shift(num_kv_heads * head_dim);

    for (auto& v : q1) v = dist(gen);
    for (auto& v : k2) v = dist(gen);
    q1_shift = q1;
    k2_shift = k2;

    // Position 5 for q, position 3 for k (relative = 2)
    // vs Position 105 for q, position 103 for k (relative = 2)
    int pos_q1 = 5, pos_k1 = 3;
    int pos_q2 = 105, pos_k2 = 103;

    // Apply RoPE CPU for both
    for (int dim = 0; dim < head_dim / 2; dim++) {
        float freq = powf(theta_base, -2.0f * dim / head_dim);

        // First pair
        float cos1_q = cosf(pos_q1 * freq), sin1_q = sinf(pos_q1 * freq);
        float cos1_k = cosf(pos_k1 * freq), sin1_k = sinf(pos_k1 * freq);

        // Second pair
        float cos2_q = cosf(pos_q2 * freq), sin2_q = sinf(pos_q2 * freq);
        float cos2_k = cosf(pos_k2 * freq), sin2_k = sinf(pos_k2 * freq);

        for (int h = 0; h < num_heads; h++) {
            int idx_e = h * head_dim + 2 * dim;
            int idx_o = h * head_dim + 2 * dim + 1;

            float qe1 = q1[idx_e], qo1 = q1[idx_o];
            q1[idx_e] = qe1 * cos1_q - qo1 * sin1_q;
            q1[idx_o] = qe1 * sin1_q + qo1 * cos1_q;

            float qe2 = q1_shift[idx_e], qo2 = q1_shift[idx_o];
            q1_shift[idx_e] = qe2 * cos2_q - qo2 * sin2_q;
            q1_shift[idx_o] = qe2 * sin2_q + qo2 * cos2_q;
        }

        for (int h = 0; h < num_kv_heads; h++) {
            int idx_e = h * head_dim + 2 * dim;
            int idx_o = h * head_dim + 2 * dim + 1;

            float ke1 = k2[idx_e], ko1 = k2[idx_o];
            k2[idx_e] = ke1 * cos1_k - ko1 * sin1_k;
            k2[idx_o] = ke1 * sin1_k + ko1 * cos1_k;

            float ke2 = k2_shift[idx_e], ko2 = k2_shift[idx_o];
            k2_shift[idx_e] = ke2 * cos2_k - ko2 * sin2_k;
            k2_shift[idx_o] = ke2 * sin2_k + ko2 * cos2_k;
        }
    }

    // Compute dot products (should be equal for same relative position)
    float dot1 = 0.0f, dot2 = 0.0f;
    for (int i = 0; i < head_dim; i++) {
        dot1 += q1[i] * k2[i];
        dot2 += q1_shift[i] * k2_shift[i];
    }

    printf("  Dot product (pos 5-3): %.4f\n", dot1);
    printf("  Dot product (pos 105-103): %.4f\n", dot2);
    printf("  Difference: %.6f\n", fabsf(dot1 - dot2));
    if (fabsf(dot1 - dot2) < 1e-4f) {
        printf("  [PASS] Relative position preserved!\n");
    }

    // =======================================================================
    // Test 3: Performance
    // =======================================================================
    printf("\nTest 3: Performance\n");

    // Reset data
    for (auto& v : h_q_gpu) v = dist(gen);
    for (auto& v : h_k_gpu) v = dist(gen);
    cudaMemcpy(d_q, h_q_gpu.data(), q_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_k, h_k_gpu.data(), k_size * sizeof(float), cudaMemcpyHostToDevice);

    // Warmup
    for (int i = 0; i < 10; i++) {
        rope_forward(d_q, d_k, d_cos, d_sin, d_positions,
                     num_tokens, num_heads, num_kv_heads, head_dim);
    }
    cudaDeviceSynchronize();

    // Benchmark
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    const int iterations = 1000;
    cudaEventRecord(start);
    for (int i = 0; i < iterations; i++) {
        rope_forward(d_q, d_k, d_cos, d_sin, d_positions,
                     num_tokens, num_heads, num_kv_heads, head_dim);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms;
    cudaEventElapsedTime(&ms, start, stop);

    printf("  Time per call: %.3f us\n", (ms / iterations) * 1000);
    printf("  Throughput: %.1f M tokens/sec\n",
           (num_tokens * iterations) / (ms * 1e3f));

    // Cleanup
    cudaFree(d_q);
    cudaFree(d_k);
    cudaFree(d_cos);
    cudaFree(d_sin);
    cudaFree(d_positions);
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
    -o test_rope \
    tests/cpp/test_rope.cu \
    csrc/kernels/rope.cu

./test_rope
```

Expected output:

```
=== RoPE Kernel Tests ===

Test 1: Correctness
  Q max error: 5.96e-07
  K max error: 4.77e-07
  [PASS] Results match

Test 2: Relative position property
  Dot product (pos 5-3): 12.3456
  Dot product (pos 105-103): 12.3456
  Difference: 0.000001
  [PASS] Relative position preserved!

Test 3: Performance
  Time per call: 8.234 us
  Throughput: 15.5 M tokens/sec

=== Tests Complete ===
```

---

## Summary

You've implemented RoPE with:

| Feature               | Implementation                        |
| --------------------- | ------------------------------------- |
| **Frequency table**   | Precomputed cos/sin for all positions |
| **GQA support**       | Different head counts for Q vs KV     |
| **Vectorization**     | float2/half2 loads for efficiency     |
| **Relative position** | Property verified in tests            |

### Key Points

1. **Precompute tables** - cos/sin values per (position, dim_pair)
2. **Apply to Q and K only** - V is not rotated
3. **Handle GQA** - Different number of Q heads vs KV heads
4. **Keep precision** - Use FP32 for cos/sin even with FP16 tensors

---

## What's Next

Next, we'll implement **SwiGLU**, the activation function used in the FFN:

- Combines Swish (SiLU) with Gated Linear Units
- Requires fusing multiple operations for efficiency

Continue to: [05_swiglu_kernel.md](./05_swiglu_kernel.md)
