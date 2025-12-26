// =============================================================================
// indexing_2d.cu - 2D Grid and Block Indexing
// =============================================================================

#include <cstdio>
#include <cuda_runtime.h>

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

__global__ void matrix_add(const float *A, const float *B, float *C, int rows,
                           int cols) {
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
  float *h_A = new float[ROWS * COLS];
  float *h_B = new float[ROWS * COLS];
  float *h_C = new float[ROWS * COLS];

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
  dim3 grid((COLS + block.x - 1) / block.x, // Number of blocks in x
            (ROWS + block.y - 1) / block.y  // Number of blocks in y
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
  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);
  delete[] h_A;
  delete[] h_B;
  delete[] h_C;

  return 0;
}
