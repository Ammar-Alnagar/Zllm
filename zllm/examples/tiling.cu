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

#include <cstdio>
#include <cuda_runtime.h>

#define TILE_SIZE 16 // Each tile is 16×16

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
__global__ void tiled_matmul(const float *A, const float *B, float *C, int M,
                             int K, int N) {
  // Shared memory for tiles of A and B
  __shared__ float tile_A[TILE_SIZE][TILE_SIZE];
  __shared__ float tile_B[TILE_SIZE][TILE_SIZE];

  // Thread indices within the block
  int tx = threadIdx.x; // Column within tile
  int ty = threadIdx.y; // Row within tile

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
      tile_A[ty][tx] = 0.0f; // Padding for out-of-bounds
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
  float *h_A = new float[M * K];
  float *h_B = new float[K * N];
  float *h_C = new float[M * N];

  // Initialize with random values
  for (int i = 0; i < M * K; i++)
    h_A[i] = static_cast<float>(rand()) / RAND_MAX;
  for (int i = 0; i < K * N; i++)
    h_B[i] = static_cast<float>(rand()) / RAND_MAX;

  // Allocate device memory
  float *d_A, *d_B, *d_C;
  cudaMalloc(&d_A, size_A);
  cudaMalloc(&d_B, size_B);
  cudaMalloc(&d_C, size_C);

  // Copy input
  cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice);

  // Launch kernel
  dim3 block(TILE_SIZE, TILE_SIZE); // 16×16 = 256 threads per block
  dim3 grid((N + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE);

  printf("Matrix multiplication: [%d×%d] × [%d×%d] = [%d×%d]\n", M, K, K, N, M,
         N);
  printf("Grid: %d×%d blocks, Block: %d×%d threads\n", grid.x, grid.y, block.x,
         block.y);

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
  float gflops =
      (2.0f * M * N * K * iterations) / (ms * 1e6); // 2 FLOPS per multiply-add
  printf("Time: %.3f ms, Performance: %.1f GFLOPS\n", ms / iterations, gflops);

  // Cleanup
  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);
  delete[] h_A;
  delete[] h_B;
  delete[] h_C;
  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  return 0;
}
