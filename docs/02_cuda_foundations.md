# Phase 1: CUDA Foundations

## Table of Contents

1. [GPU Architecture Overview](#gpu-architecture-overview)
2. [CUDA Programming Model](#cuda-programming-model)
3. [Memory Hierarchy](#memory-hierarchy)
4. [Thread Organization](#thread-organization)
5. [Memory Coalescing](#memory-coalescing)
6. [Warp-Level Programming](#warp-level-programming)
7. [Shared Memory Patterns](#shared-memory-patterns)
8. [Practical Examples](#practical-examples)

---

## GPU Architecture Overview

Before writing kernels, understand how modern NVIDIA GPUs are organized:

```
                           GPU Architecture (Ampere/Ada)
┌──────────────────────────────────────────────────────────────────────────────┐
│                              GPU Chip                                        │
│  ┌────────────────────────────────────────────────────────────────────────┐  │
│  │                        L2 Cache (40-96 MB)                             │  │
│  └────────────────────────────────────────────────────────────────────────┘  │
│                                    │                                          │
│     ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐     (up to 128 SMs)  │
│     │   SM 0   │ │   SM 1   │ │   SM 2   │ │   SM 3   │ ...                  │
│     │          │ │          │ │          │ │          │                       │
│     │ ┌──────┐ │ │ ┌──────┐ │ │ ┌──────┐ │ │ ┌──────┐ │                       │
│     │ │L1/SM │ │ │ │L1/SM │ │ │ │L1/SM │ │ │ │L1/SM │ │  (192KB L1+Shared)   │
│     │ │Cache │ │ │ │Cache │ │ │ │Cache │ │ │ │Cache │ │                       │
│     │ └──────┘ │ │ └──────┘ │ │ └──────┘ │ │ └──────┘ │                       │
│     │          │ │          │ │          │ │          │                       │
│     │ Warps:   │ │ Warps:   │ │ Warps:   │ │ Warps:   │                       │
│     │ 32 thds  │ │ 32 thds  │ │ 32 thds  │ │ 32 thds  │  (64 warps/SM max)   │
│     │ ×64warps │ │ ×64warps │ │ ×64warps │ │ ×64warps │                       │
│     │          │ │          │ │          │ │          │                       │
│     │ Tensor   │ │ Tensor   │ │ Tensor   │ │ Tensor   │  (4 Tensor Cores/SM)  │
│     │ Cores    │ │ Cores    │ │ Cores    │ │ Cores    │                       │
│     └──────────┘ └──────────┘ └──────────┘ └──────────┘                       │
│                                    │                                          │
│  ┌────────────────────────────────────────────────────────────────────────┐  │
│  │                    HBM (High Bandwidth Memory)                         │  │
│  │                       16-80 GB @ 1-3 TB/s                              │  │
│  └────────────────────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────────────────┘
```

### Key Components

| Component           | RTX 4090     | A100        | H100       |
| ------------------- | ------------ | ----------- | ---------- |
| **SMs**             | 128          | 108         | 132        |
| **CUDA Cores/SM**   | 128          | 64          | 128        |
| **Tensor Cores/SM** | 4            | 4           | 4          |
| **L2 Cache**        | 72 MB        | 40 MB       | 50 MB      |
| **Memory**          | 24 GB GDDR6X | 80 GB HBM2e | 80 GB HBM3 |
| **Bandwidth**       | 1 TB/s       | 2 TB/s      | 3.35 TB/s  |
| **Shared Mem/SM**   | 100 KB       | 164 KB      | 228 KB     |

---

## CUDA Programming Model

### Hello World Kernel

Let's start with a simple kernel to understand the basics:

Create file: `mini_vllm/examples/hello_cuda.cu`

```c++
// =============================================================================
// hello_cuda.cu - Your First CUDA Kernel
// =============================================================================
// This example demonstrates the basic structure of a CUDA program:
// 1. Define a kernel function (__global__)
// 2. Allocate GPU memory
// 3. Copy data to GPU
// 4. Launch kernel
// 5. Copy results back
// 6. Free memory
// =============================================================================

#include <cuda_runtime.h>
#include <cstdio>

// =============================================================================
// Kernel Definition
// =============================================================================
// __global__ indicates this function runs on GPU, called from CPU
// __device__ would mean runs on GPU, called from GPU
// __host__ means runs on CPU, called from CPU (default)

/**
 * vector_add - Add two vectors element-wise
 *
 * Each thread handles one element of the output array.
 *
 * @param a: First input vector (device pointer)
 * @param b: Second input vector (device pointer)
 * @param c: Output vector (device pointer)
 * @param n: Number of elements
 */
__global__ void vector_add(const float* a, const float* b, float* c, int n) {
    // Calculate global thread index
    // blockIdx.x = which block (0 to num_blocks-1)
    // blockDim.x = threads per block
    // threadIdx.x = thread index within block (0 to blockDim.x-1)
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Bounds check - important when n is not divisible by block size
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

// =============================================================================
// Main Function (Host Code)
// =============================================================================

int main() {
    // Vector size
    const int N = 1024 * 1024;  // 1M elements
    const size_t size = N * sizeof(float);

    printf("Vector Addition: %d elements\n", N);

    // =========================================================================
    // Step 1: Allocate host (CPU) memory
    // =========================================================================
    float* h_a = new float[N];
    float* h_b = new float[N];
    float* h_c = new float[N];

    // Initialize with some values
    for (int i = 0; i < N; i++) {
        h_a[i] = static_cast<float>(i);
        h_b[i] = static_cast<float>(i * 2);
    }

    // =========================================================================
    // Step 2: Allocate device (GPU) memory
    // =========================================================================
    float *d_a, *d_b, *d_c;

    cudaMalloc(&d_a, size);  // Returns pointer to GPU memory
    cudaMalloc(&d_b, size);
    cudaMalloc(&d_c, size);

    // =========================================================================
    // Step 3: Copy input data from host to device
    // =========================================================================
    // cudaMemcpyHostToDevice = CPU -> GPU
    // cudaMemcpyDeviceToHost = GPU -> CPU
    // cudaMemcpyDeviceToDevice = GPU -> GPU

    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);

    // =========================================================================
    // Step 4: Launch kernel
    // =========================================================================
    // Grid and block dimensions
    int threads_per_block = 256;  // Common choice: 128, 256, 512
    int num_blocks = (N + threads_per_block - 1) / threads_per_block;  // Ceiling division

    printf("Launching kernel: %d blocks × %d threads\n", num_blocks, threads_per_block);

    // Launch syntax: kernel<<<grid, block>>>(args...)
    // - grid: number of blocks (can be dim3 for 2D/3D)
    // - block: threads per block (can be dim3)
    vector_add<<<num_blocks, threads_per_block>>>(d_a, d_b, d_c, N);

    // Check for launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Kernel launch error: %s\n", cudaGetErrorString(err));
        return 1;
    }

    // =========================================================================
    // Step 5: Copy results back to host
    // =========================================================================
    cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost);

    // =========================================================================
    // Step 6: Verify results
    // =========================================================================
    bool correct = true;
    for (int i = 0; i < N && i < 10; i++) {
        float expected = h_a[i] + h_b[i];
        if (h_c[i] != expected) {
            printf("Error at %d: %f != %f\n", i, h_c[i], expected);
            correct = false;
            break;
        }
        printf("c[%d] = %f + %f = %f ✓\n", i, h_a[i], h_b[i], h_c[i]);
    }

    if (correct) {
        printf("\nAll results correct!\n");
    }

    // =========================================================================
    // Step 7: Free memory
    // =========================================================================
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    delete[] h_a;
    delete[] h_b;
    delete[] h_c;

    return 0;
}
```

Compile and run:

```bash
nvcc -o hello_cuda examples/hello_cuda.cu
./hello_cuda
```

Expected output:

```
Vector Addition: 1048576 elements
Launching kernel: 4096 blocks × 256 threads
c[0] = 0.000000 + 0.000000 = 0.000000 ✓
c[1] = 1.000000 + 2.000000 = 3.000000 ✓
c[2] = 2.000000 + 4.000000 = 6.000000 ✓
...
All results correct!
```

---

## Memory Hierarchy

Understanding GPU memory is critical for performance:

```
                    GPU Memory Hierarchy

Speed:  FASTEST ─────────────────────────────────▶ SLOWEST

        ┌─────────────────────┐
        │     Registers       │  ← Per-thread, ~255 per thread
        │   (< 1 cycle)       │     Used for local variables
        └──────────┬──────────┘
                   │
        ┌──────────▼──────────┐
        │   Shared Memory     │  ← Per-block, 48-228 KB
        │   (~20 cycles)      │     Explicitly managed by programmer
        └──────────┬──────────┘
                   │
        ┌──────────▼──────────┐
        │     L1 Cache        │  ← Per-SM, 128-256 KB
        │   (~30 cycles)      │     Automatic, combined with shared
        └──────────┬──────────┘
                   │
        ┌──────────▼──────────┐
        │     L2 Cache        │  ← Shared by all SMs, 40-96 MB
        │   (~200 cycles)     │     Automatic
        └──────────┬──────────┘
                   │
        ┌──────────▼──────────┐
        │   Global Memory     │  ← Main GPU memory, 16-80 GB
        │   (HBM/GDDR6X)      │     High bandwidth, high latency
        │   (~400 cycles)     │
        └─────────────────────┘
```

### Memory Types in CUDA

```c++
// =============================================================================
// memory_types.cu - Demonstrating GPU Memory Types
// =============================================================================

#include <cuda_runtime.h>
#include <cstdio>

// -----------------------------------------------------------------------------
// 1. Global Memory - Accessible by all threads, persists for kernel lifetime
// -----------------------------------------------------------------------------
// Declared with __device__ at file scope
__device__ float d_global_array[1024];  // Static allocation

// Dynamic allocation uses cudaMalloc (shown in main)

// -----------------------------------------------------------------------------
// 2. Constant Memory - Read-only, cached, broadcast to all threads
// -----------------------------------------------------------------------------
// Great for values read by all threads (e.g., model hyperparameters)
__constant__ float d_constants[256];  // Max 64KB constant memory

// -----------------------------------------------------------------------------
// 3. Shared Memory - Per-block, extremely fast
// -----------------------------------------------------------------------------
// Static: declared with __shared__ inside kernel
// Dynamic: declared extern __shared__ and size specified at launch

__global__ void memory_demo(float* global_out, int n) {
    // -------------------------------------------------------------------------
    // Registers - fastest, per-thread
    // -------------------------------------------------------------------------
    float local_var = 0.0f;  // Stored in register
    int thread_idx = threadIdx.x;  // Also register

    // -------------------------------------------------------------------------
    // Shared Memory - Static Allocation
    // -------------------------------------------------------------------------
    // All 256 threads in the block share this array
    __shared__ float shared_static[256];

    // -------------------------------------------------------------------------
    // Shared Memory - Dynamic Allocation
    // -------------------------------------------------------------------------
    // Size specified at kernel launch: kernel<<<grid, block, shared_size>>>()
    extern __shared__ float shared_dynamic[];

    // -------------------------------------------------------------------------
    // Example: Using shared memory for reduction
    // -------------------------------------------------------------------------

    // Each thread initializes its slot
    shared_static[thread_idx] = (float)thread_idx;

    // CRITICAL: Sync threads before reading shared memory
    // Without this, some threads might read before others write!
    __syncthreads();

    // Now all threads can safely read any element
    local_var = shared_static[(thread_idx + 1) % 256];

    // Write to global memory
    int global_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (global_idx < n) {
        global_out[global_idx] = local_var;
    }
}

// -----------------------------------------------------------------------------
// Main
// -----------------------------------------------------------------------------
int main() {
    const int N = 1024;
    const size_t size = N * sizeof(float);

    // Allocate global memory dynamically
    float* d_out;
    cudaMalloc(&d_out, size);

    // Copy to constant memory
    float h_constants[256];
    for (int i = 0; i < 256; i++) {
        h_constants[i] = (float)i * 0.1f;
    }
    cudaMemcpyToSymbol(d_constants, h_constants, 256 * sizeof(float));

    // Launch kernel with dynamic shared memory
    int threads = 256;
    int blocks = N / threads;
    size_t shared_size = threads * sizeof(float);  // Dynamic shared size

    memory_demo<<<blocks, threads, shared_size>>>(d_out, N);
    cudaDeviceSynchronize();

    // Verify
    float* h_out = new float[N];
    cudaMemcpy(h_out, d_out, size, cudaMemcpyDeviceToHost);

    printf("First 10 results:\n");
    for (int i = 0; i < 10; i++) {
        printf("  out[%d] = %f\n", i, h_out[i]);
    }

    cudaFree(d_out);
    delete[] h_out;

    return 0;
}
```

---

## Thread Organization

### Grid, Block, and Thread Hierarchy

```
                         CUDA Thread Hierarchy

    Grid (entire kernel launch)
    ┌─────────────────────────────────────────────────────────────────┐
    │                                                                 │
    │   Block (0,0)         Block (1,0)         Block (2,0)           │
    │   ┌───────────────┐   ┌───────────────┐   ┌───────────────┐     │
    │   │ Thread Thread │   │ Thread Thread │   │ Thread Thread │     │
    │   │ (0,0)  (1,0)  │   │ (0,0)  (1,0)  │   │ (0,0)  (1,0)  │     │
    │   │               │   │               │   │               │     │
    │   │ Thread Thread │   │ Thread Thread │   │ Thread Thread │     │
    │   │ (0,1)  (1,1)  │   │ (0,1)  (1,1)  │   │ (0,1)  (1,1)  │     │
    │   └───────────────┘   └───────────────┘   └───────────────┘     │
    │                                                                 │
    │   Block (0,1)         Block (1,1)         Block (2,1)           │
    │   ┌───────────────┐   ┌───────────────┐   ┌───────────────┐     │
    │   │ Thread Thread │   │ Thread Thread │   │ Thread Thread │     │
    │   │ (0,0)  (1,0)  │   │ (0,0)  (1,0)  │   │ (0,0)  (1,0)  │     │
    │   │               │   │               │   │               │     │
    │   │ Thread Thread │   │ Thread Thread │   │ Thread Thread │     │
    │   │ (0,1)  (1,1)  │   │ (0,1)  (1,1)  │   │ (0,1)  (1,1)  │     │
    │   └───────────────┘   └───────────────┘   └───────────────┘     │
    │                                                                 │
    └─────────────────────────────────────────────────────────────────┘

    gridDim.x = 3, gridDim.y = 2       (6 blocks total)
    blockDim.x = 2, blockDim.y = 2     (4 threads per block)

    Built-in variables:
    - gridDim: dimensions of the grid (number of blocks)
    - blockDim: dimensions of each block (threads per block)
    - blockIdx: this block's index in the grid
    - threadIdx: this thread's index in the block
```

### 2D Indexing Example

```c++
// =============================================================================
// indexing_2d.cu - 2D Grid and Block Indexing
// =============================================================================

#include <cuda_runtime.h>
#include <cstdio>

/**
 * For processing 2D data like matrices or images, we use 2D grids and blocks.
 *
 * Matrix layout (row-major):
 *
 *        Col 0   Col 1   Col 2   Col 3
 *      ┌───────┬───────┬───────┬───────┐
 * Row 0│ M[0,0]│ M[0,1]│ M[0,2]│ M[0,3]│
 *      ├───────┼───────┼───────┼───────┤
 * Row 1│ M[1,0]│ M[1,1]│ M[1,2]│ M[1,3]│
 *      ├───────┼───────┼───────┼───────┤
 * Row 2│ M[2,0]│ M[2,1]│ M[2,2]│ M[2,3]│
 *      └───────┴───────┴───────┴───────┘
 *
 * In memory: [M[0,0], M[0,1], M[0,2], M[0,3], M[1,0], M[1,1], ...]
 * Index: row * num_cols + col
 */

__global__ void matrix_add(const float* A, const float* B, float* C,
                           int rows, int cols) {
    // Calculate row and column for this thread
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // Bounds check
    if (row < rows && col < cols) {
        // Linear index in row-major layout
        int idx = row * cols + col;
        C[idx] = A[idx] + B[idx];
    }
}

int main() {
    const int ROWS = 1024;
    const int COLS = 2048;
    const size_t size = ROWS * COLS * sizeof(float);

    // Allocate and initialize host memory
    float* h_A = new float[ROWS * COLS];
    float* h_B = new float[ROWS * COLS];
    float* h_C = new float[ROWS * COLS];

    for (int i = 0; i < ROWS * COLS; i++) {
        h_A[i] = 1.0f;
        h_B[i] = 2.0f;
    }

    // Allocate device memory
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);

    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    // Configure 2D grid and blocks
    // Each block has 16×16 = 256 threads (good occupancy)
    dim3 block(16, 16);

    // Calculate grid size to cover entire matrix
    // Use ceiling division to handle non-divisible dimensions
    dim3 grid(
        (COLS + block.x - 1) / block.x,  // Number of blocks in x
        (ROWS + block.y - 1) / block.y   // Number of blocks in y
    );

    printf("Matrix: %d × %d\n", ROWS, COLS);
    printf("Block: %d × %d threads\n", block.x, block.y);
    printf("Grid: %d × %d blocks\n", grid.x, grid.y);
    printf("Total threads: %d\n", grid.x * grid.y * block.x * block.y);

    // Launch kernel
    matrix_add<<<grid, block>>>(d_A, d_B, d_C, ROWS, COLS);
    cudaDeviceSynchronize();

    // Copy and verify
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    bool correct = true;
    for (int i = 0; i < ROWS * COLS; i++) {
        if (h_C[i] != 3.0f) {
            printf("Error at index %d: %f != 3.0\n", i, h_C[i]);
            correct = false;
            break;
        }
    }

    if (correct) {
        printf("All results correct!\n");
    }

    // Cleanup
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    delete[] h_A; delete[] h_B; delete[] h_C;

    return 0;
}
```

---

## Memory Coalescing

**Memory coalescing** is one of the most critical optimizations:

```
                    Memory Coalescing

COALESCED ACCESS (adjacent threads access adjacent memory):

Thread 0   Thread 1   Thread 2   Thread 3   ...   Thread 31
   │          │          │          │                 │
   ▼          ▼          ▼          ▼                 ▼
┌──────────────────────────────────────────────────────────┐
│  M[0]  │  M[1]  │  M[2]  │  M[3]  │  ...  │  M[31] │    │  ← Single 128-byte transaction
└──────────────────────────────────────────────────────────┘

Result: 1 memory transaction serves all 32 threads!


UNCOALESCED ACCESS (strided access pattern):

Thread 0   Thread 1   Thread 2   Thread 3
   │          │          │          │
   ▼          │          │          │
┌──────┐      │          │          │
│ M[0] │      │          │          │
└──────┘      ▼          │          │
              ┌──────┐   │          │
              │M[128]│   │          │
              └──────┘   ▼          │
                         ┌──────┐   │
                         │M[256]│   │
                         └──────┘   ▼
                                    ┌──────┐
                                    │M[384]│
                                    └──────┘

Result: 4 separate memory transactions! (stride = 128 elements)
```

### Coalescing Example

```c++
// =============================================================================
// coalescing.cu - Memory Coalescing Demonstration
// =============================================================================

#include <cuda_runtime.h>
#include <cstdio>

// Good: Coalesced access
__global__ void coalesced_read(const float* input, float* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        // Adjacent threads read adjacent memory locations
        // Thread 0 reads input[0], Thread 1 reads input[1], etc.
        output[idx] = input[idx] * 2.0f;
    }
}

// Bad: Strided access (uncoalesced)
__global__ void strided_read(const float* input, float* output, int n, int stride) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    // Thread 0 reads input[0], Thread 1 reads input[stride], etc.
    // This causes multiple memory transactions!
    int src_idx = idx * stride;
    if (src_idx < n) {
        output[idx] = input[src_idx] * 2.0f;
    }
}

// Benchmark helper
void benchmark(const char* name, void (*kernel)(const float*, float*, int),
               float* d_in, float* d_out, int n, int iterations) {
    // Warmup
    kernel<<<n/256, 256>>>(d_in, d_out, n);
    cudaDeviceSynchronize();

    // Time
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    for (int i = 0; i < iterations; i++) {
        kernel<<<n/256, 256>>>(d_in, d_out, n);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms;
    cudaEventElapsedTime(&ms, start, stop);

    float gb = (2.0f * n * sizeof(float) * iterations) / 1e9;  // Read + Write
    printf("%s: %.2f ms, %.1f GB/s\n", name, ms/iterations, gb / (ms/1000));

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

int main() {
    const int N = 64 * 1024 * 1024;  // 64M elements
    const size_t size = N * sizeof(float);

    float* d_in;
    float* d_out;
    cudaMalloc(&d_in, size);
    cudaMalloc(&d_out, size);

    // Initialize
    float* h_data = new float[N];
    for (int i = 0; i < N; i++) h_data[i] = (float)i;
    cudaMemcpy(d_in, h_data, size, cudaMemcpyHostToDevice);

    printf("Benchmarking memory access patterns (%d elements):\n\n", N);

    // Benchmark coalesced
    benchmark("Coalesced   ",
              [](const float* in, float* out, int n) {
                  coalesced_read<<<n/256, 256>>>(in, out, n);
              },
              d_in, d_out, N, 100);

    // Benchmark strided (stride=32 is worst case for warp)
    printf("\n");
    benchmark("Strided (32)",
              [](const float* in, float* out, int n) {
                  strided_read<<<n/256, 256>>>(in, out, n/32, 32);
              },
              d_in, d_out, N, 100);

    cudaFree(d_in);
    cudaFree(d_out);
    delete[] h_data;

    return 0;
}
```

---

## Warp-Level Programming

A **warp** is a group of 32 threads that execute in lockstep. Understanding warps is essential for high-performance kernels.

```
                         Warp Execution Model

Block with 128 threads = 4 warps:

Warp 0: Threads 0-31    ──┬── Execute same instruction
Warp 1: Threads 32-63   ──┼── at the same time
Warp 2: Threads 64-95   ──┤   (SIMT: Single Instruction
Warp 3: Threads 96-127  ──┘    Multiple Threads)

Within a warp, threads can communicate WITHOUT shared memory!
Use warp shuffle instructions: __shfl_sync, __shfl_down_sync, etc.
```

### Warp Shuffle Instructions

```c++
// =============================================================================
// warp_primitives.cu - Warp-Level Programming
// =============================================================================

#include <cuda_runtime.h>
#include <cstdio>

constexpr int WARP_SIZE = 32;

/**
 * Warp Reduction using Shuffle
 *
 * Much faster than shared memory for intra-warp communication!
 *
 * Shuffle Down Pattern for Reduction:
 *
 * Initial values (each thread has one value):
 * Lane:  0  1  2  3  4  5  6  7  ...  31
 * Value: a  b  c  d  e  f  g  h  ...  z
 *
 * After __shfl_down with offset=16:
 * Lane:  0       1       ...  15      | 16 17 ... 31
 * Gets:  a+lane16 b+lane17...  p+z    | (unchanged)
 *
 * After offset=8: lanes 0-7 have sum of lanes 0-7 + 8-15 + 16-23 + 24-31
 * After offset=4: lanes 0-3 have sum of lanes 0-15 + 16-31
 * After offset=2: lanes 0-1 have sum of lanes 0-31
 * After offset=1: lane 0 has sum of all 32 lanes!
 */
__device__ float warp_reduce_sum(float val) {
    // 0xffffffff = all 32 lanes participate
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;  // Only lane 0 has the final sum
}

/**
 * Warp Broadcast - Send value from one lane to all others
 */
__device__ float warp_broadcast(float val, int src_lane) {
    return __shfl_sync(0xffffffff, val, src_lane);
}

/**
 * Warp Scan (Prefix Sum) - Each lane gets sum of all previous lanes
 *
 * Input:  1  2  3  4  5  6  7  8  ...
 * Output: 1  3  6  10 15 21 28 36 ... (cumulative sum)
 */
__device__ float warp_prefix_sum(float val) {
    #pragma unroll
    for (int offset = 1; offset < WARP_SIZE; offset *= 2) {
        float n = __shfl_up_sync(0xffffffff, val, offset);
        if (threadIdx.x % WARP_SIZE >= offset) {
            val += n;
        }
    }
    return val;
}

// Demo kernel
__global__ void warp_demo(float* output) {
    int lane = threadIdx.x % WARP_SIZE;
    int warp_id = threadIdx.x / WARP_SIZE;

    // Each thread starts with its lane id as value
    float my_value = (float)lane;

    // Reduce
    float sum = warp_reduce_sum(my_value);

    // Only lane 0 writes the result
    if (lane == 0) {
        output[warp_id] = sum;
        printf("Warp %d: sum of lanes 0-31 = %.0f (expected: %d)\n",
               warp_id, sum, 31 * 32 / 2);  // Sum of 0..31 = 496
    }
}

int main() {
    float* d_output;
    cudaMalloc(&d_output, 4 * sizeof(float));

    // Launch with 128 threads = 4 warps
    warp_demo<<<1, 128>>>(d_output);
    cudaDeviceSynchronize();

    float h_output[4];
    cudaMemcpy(h_output, d_output, 4 * sizeof(float), cudaMemcpyDeviceToHost);

    printf("\nResults:\n");
    for (int i = 0; i < 4; i++) {
        printf("  Warp %d result: %.0f\n", i, h_output[i]);
    }

    cudaFree(d_output);
    return 0;
}
```

---

## Shared Memory Patterns

### Tiled Matrix Multiplication

This is the foundation for Flash Attention!

```c++
// =============================================================================
// tiled_matmul.cu - Tiled Matrix Multiplication
// =============================================================================
//
// Matrix multiplication: C = A × B
//
// Naive approach: Each thread computes one element of C
// - Reads entire row of A and column of B
// - O(N) global memory reads per output element
// - Very inefficient!
//
// Tiled approach: Load tiles into shared memory
// - Each block works on a tile of C
// - Loads tiles of A and B into shared memory
// - Much fewer global memory accesses
// =============================================================================

#include <cuda_runtime.h>
#include <cstdio>

#define TILE_SIZE 16  // Each tile is 16×16

/**
 * tiled_matmul - Matrix multiplication using shared memory tiling
 *
 * C[M×N] = A[M×K] × B[K×N]
 *
 * Visualization of tiling:
 *
 *     Matrix A (M×K)          Matrix B (K×N)           Matrix C (M×N)
 *     ┌───────────────┐       ┌───────────────┐       ┌───────────────┐
 *     │░░░│   │   │   │       │░░░│   │   │   │       │░░░│   │   │   │
 *     │░░░│   │   │   │       │░░░│   │   │   │       │───┼───┼───┼───│
 *   ▲ │───│───│───│───│     ▲ │░░░│   │   │   │       │   │   │   │   │
 *   M │   │   │   │   │     K │░░░│   │   │   │       │   │   │   │   │
 *     │───│───│───│───│       │───│───│───│───│       │───│───│───│───│
 *     │   │   │   │   │       │   │   │   │   │       │   │   │   │   │
 *     └───────────────┘       └───────────────┘       └───────────────┘
 *           K ──▶                    N ──▶                   N ──▶
 *
 *     One block computes one tile (░░░) of C
 *     It iterates through tiles of A and B
 */
__global__ void tiled_matmul(const float* A, const float* B, float* C,
                              int M, int K, int N) {
    // Shared memory for tiles of A and B
    __shared__ float tile_A[TILE_SIZE][TILE_SIZE];
    __shared__ float tile_B[TILE_SIZE][TILE_SIZE];

    // Thread indices within the block
    int tx = threadIdx.x;  // Column within tile
    int ty = threadIdx.y;  // Row within tile

    // Output position this thread is responsible for
    int row = blockIdx.y * TILE_SIZE + ty;
    int col = blockIdx.x * TILE_SIZE + tx;

    // Accumulator for dot product
    float sum = 0.0f;

    // Number of tiles to iterate over
    int num_tiles = (K + TILE_SIZE - 1) / TILE_SIZE;

    for (int t = 0; t < num_tiles; t++) {
        // =================================================================
        // Phase 1: Load tiles into shared memory (cooperative loading)
        // =================================================================

        // Load one element of A tile
        // Row: row (same as output row)
        // Col: t * TILE_SIZE + tx (iterating through K dimension)
        int a_col = t * TILE_SIZE + tx;
        if (row < M && a_col < K) {
            tile_A[ty][tx] = A[row * K + a_col];
        } else {
            tile_A[ty][tx] = 0.0f;  // Padding for out-of-bounds
        }

        // Load one element of B tile
        // Row: t * TILE_SIZE + ty (iterating through K dimension)
        // Col: col (same as output col)
        int b_row = t * TILE_SIZE + ty;
        if (b_row < K && col < N) {
            tile_B[ty][tx] = B[b_row * N + col];
        } else {
            tile_B[ty][tx] = 0.0f;
        }

        // Wait for all threads to finish loading
        __syncthreads();

        // =================================================================
        // Phase 2: Compute partial dot product using tiles
        // =================================================================
        #pragma unroll
        for (int k = 0; k < TILE_SIZE; k++) {
            sum += tile_A[ty][k] * tile_B[k][tx];
        }

        // Wait before loading next tiles (don't overwrite while others read)
        __syncthreads();
    }

    // Write result
    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

int main() {
    const int M = 512;
    const int K = 512;
    const int N = 512;

    const size_t size_A = M * K * sizeof(float);
    const size_t size_B = K * N * sizeof(float);
    const size_t size_C = M * N * sizeof(float);

    // Allocate host memory
    float* h_A = new float[M * K];
    float* h_B = new float[K * N];
    float* h_C = new float[M * N];

    // Initialize with random values
    for (int i = 0; i < M * K; i++) h_A[i] = static_cast<float>(rand()) / RAND_MAX;
    for (int i = 0; i < K * N; i++) h_B[i] = static_cast<float>(rand()) / RAND_MAX;

    // Allocate device memory
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size_A);
    cudaMalloc(&d_B, size_B);
    cudaMalloc(&d_C, size_C);

    // Copy input
    cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice);

    // Launch kernel
    dim3 block(TILE_SIZE, TILE_SIZE);  // 16×16 = 256 threads per block
    dim3 grid((N + TILE_SIZE - 1) / TILE_SIZE,
              (M + TILE_SIZE - 1) / TILE_SIZE);

    printf("Matrix multiplication: [%d×%d] × [%d×%d] = [%d×%d]\n", M, K, K, N, M, N);
    printf("Grid: %d×%d blocks, Block: %d×%d threads\n",
           grid.x, grid.y, block.x, block.y);

    // Warmup
    tiled_matmul<<<grid, block>>>(d_A, d_B, d_C, M, K, N);
    cudaDeviceSynchronize();

    // Benchmark
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    const int iterations = 100;
    cudaEventRecord(start);
    for (int i = 0; i < iterations; i++) {
        tiled_matmul<<<grid, block>>>(d_A, d_B, d_C, M, K, N);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms;
    cudaEventElapsedTime(&ms, start, stop);

    // Calculate performance
    float gflops = (2.0f * M * N * K * iterations) / (ms * 1e6);  // 2 FLOPS per multiply-add
    printf("Time: %.3f ms, Performance: %.1f GFLOPS\n", ms / iterations, gflops);

    // Cleanup
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    delete[] h_A; delete[] h_B; delete[] h_C;
    cudaEventDestroy(start); cudaEventDestroy(stop);

    return 0;
}
```

---

## Practical Examples

### Block-Level Reduction

```c++
// =============================================================================
// block_reduction.cu - Complete Block Reduction Pattern
// =============================================================================
// This pattern is used extensively in RMSNorm, Softmax, and other operations.
// =============================================================================

#include <cuda_runtime.h>
#include <cstdio>

constexpr int WARP_SIZE = 32;
constexpr int MAX_BLOCK_SIZE = 1024;

/**
 * Block reduction: Sum all input elements within a block
 *
 * Uses two-level hierarchy:
 * 1. Warp-level reduction using shuffle (no sync needed!)
 * 2. Cross-warp reduction using shared memory
 */
template<int BLOCK_SIZE>
__device__ float block_reduce_sum(float val) {
    // Shared memory for warp partial sums
    // We need one slot per warp
    constexpr int NUM_WARPS = BLOCK_SIZE / WARP_SIZE;
    __shared__ float warp_sums[NUM_WARPS];

    int lane = threadIdx.x % WARP_SIZE;
    int warp_id = threadIdx.x / WARP_SIZE;

    // Step 1: Reduce within warp
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }

    // Step 2: First lane of each warp writes to shared memory
    if (lane == 0) {
        warp_sums[warp_id] = val;
    }
    __syncthreads();

    // Step 3: First warp reduces warp sums
    if (warp_id == 0) {
        val = (lane < NUM_WARPS) ? warp_sums[lane] : 0.0f;

        #pragma unroll
        for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
            val += __shfl_down_sync(0xffffffff, val, offset);
        }
    }

    // Broadcast result to all threads
    if (threadIdx.x == 0) {
        warp_sums[0] = val;
    }
    __syncthreads();

    return warp_sums[0];
}

// Example usage: Compute sum of an array
__global__ void array_sum(const float* input, float* output, int n) {
    // Each block processes a portion of the input
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Load value (or 0 if out of bounds)
    float val = (idx < n) ? input[idx] : 0.0f;

    // Reduce within block
    float block_sum = block_reduce_sum<256>(val);

    // Thread 0 writes block result
    if (threadIdx.x == 0) {
        atomicAdd(output, block_sum);
    }
}

int main() {
    const int N = 1024 * 1024;

    // Allocate
    float* h_input = new float[N];
    float h_output = 0.0f;
    float* d_input;
    float* d_output;

    cudaMalloc(&d_input, N * sizeof(float));
    cudaMalloc(&d_output, sizeof(float));

    // Initialize with 1s (sum should be N)
    for (int i = 0; i < N; i++) h_input[i] = 1.0f;
    cudaMemcpy(d_input, h_input, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(d_output, 0, sizeof(float));

    // Launch
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    array_sum<<<blocks, threads>>>(d_input, d_output, N);

    // Get result
    cudaMemcpy(&h_output, d_output, sizeof(float), cudaMemcpyDeviceToHost);

    printf("Sum of %d ones = %.0f (expected: %d)\n", N, h_output, N);

    // Cleanup
    cudaFree(d_input);
    cudaFree(d_output);
    delete[] h_input;

    return 0;
}
```

---

## Summary

You've learned the essential CUDA concepts:

| Concept              | Key Takeaway                                    |
| -------------------- | ----------------------------------------------- |
| **Thread Hierarchy** | Grid → Block → Warp → Thread                    |
| **Memory Types**     | Registers > Shared > L1 > L2 > Global           |
| **Coalescing**       | Adjacent threads should access adjacent memory  |
| **Warp Primitives**  | Use `__shfl_*` for intra-warp communication     |
| **Shared Memory**    | Use for inter-thread communication within block |
| **Tiling**           | Load chunks into fast memory, compute, repeat   |

### Performance Tips

1. **Maximize occupancy** - Aim for 50%+ occupancy, balance threads vs. registers
2. **Coalesce memory** - Adjacent threads read adjacent addresses
3. **Minimize syncs** - `__syncthreads()` is expensive, use warp shuffles when possible
4. **Vectorize loads** - Use `float4` for 128-bit wide loads
5. **Hide latency** - Launch enough threads to keep SMs busy during memory waits

---

## What's Next

Now that you understand CUDA fundamentals, we'll implement our first real kernel:
**RMSNorm** - Root Mean Square Normalization

This kernel demonstrates:

- Block reduction patterns
- Memory coalescing
- Warp-level optimization

Continue to: [03_rmsnorm_kernel.md](./03_rmsnorm_kernel.md)
