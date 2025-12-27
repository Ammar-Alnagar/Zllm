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

#include <cstdio>
#include <cuda_runtime.h>

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
__global__ void vector_add(const float *a, const float *b, float *c, int n) {
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
  const int N = 1024 * 1024; // 1M elements
  const size_t size = N * sizeof(float);

  printf("Vector Addition: %d elements\n", N);

  // =========================================================================
  // Step 1: Allocate host (CPU) memory
  // =========================================================================
  float *h_a = new float[N];
  float *h_b = new float[N];
  float *h_c = new float[N];

  // Initialize with some values
  for (int i = 0; i < N; i++) {
    h_a[i] = static_cast<float>(i);
    h_b[i] = static_cast<float>(i * 2);
  }

  // =========================================================================
  // Step 2: Allocate device (GPU) memory
  // =========================================================================
  float *d_a, *d_b, *d_c;

  cudaMalloc(&d_a, size); // Returns pointer to GPU memory
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
  int threads_per_block = 512; // Common choice: 128, 256, 512
  int num_blocks =
      (N + threads_per_block - 1) / threads_per_block; // Ceiling division

  printf("Launching kernel: %d blocks × %d threads\n", num_blocks,
         threads_per_block);

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
