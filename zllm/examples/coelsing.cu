// =============================================================================
// coalescing.cu - Memory Coalescing Demonstration
// =============================================================================

#include <cstdio>
#include <cuda_runtime.h>

// Good: Coalesced access
__global__ void coalesced_read(const float *input, float *output, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    // Adjacent threads read adjacent memory locations
    // Thread 0 reads input[0], Thread 1 reads input[1], etc.
    output[idx] = input[idx] * 2.0f;
  }
}

// Bad: Strided access (uncoalesced)
__global__ void strided_read(const float *input, float *output, int n,
                             int stride) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  // Thread 0 reads input[0], Thread 1 reads input[stride], etc.
  // This causes multiple memory transactions!
  int src_idx = idx * stride;
  if (src_idx < n) {
    output[idx] = input[src_idx] * 2.0f;
  }
}

// Benchmark helper
void benchmark(const char *name, void (*kernel)(const float *, float *, int),
               float *d_in, float *d_out, int n, int iterations) {
  // Warmup
  kernel<<<n / 256, 256>>>(d_in, d_out, n);
  cudaDeviceSynchronize();

  // Time
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start);
  for (int i = 0; i < iterations; i++) {
    kernel<<<n / 256, 256>>>(d_in, d_out, n);
  }
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);

  float ms;
  cudaEventElapsedTime(&ms, start, stop);

  float gb = (2.0f * n * sizeof(float) * iterations) / 1e9; // Read + Write
  printf("%s: %.2f ms, %.1f GB/s\n", name, ms / iterations, gb / (ms / 1000));

  cudaEventDestroy(start);
  cudaEventDestroy(stop);
}

int main() {
  const int N = 64 * 1024 * 1024; // 64M elements
  const size_t size = N * sizeof(float);

  float *d_in;
  float *d_out;
  cudaMalloc(&d_in, size);
  cudaMalloc(&d_out, size);

  // Initialize
  float *h_data = new float[N];
  for (int i = 0; i < N; i++)
    h_data[i] = (float)i;
  cudaMemcpy(d_in, h_data, size, cudaMemcpyHostToDevice);

  printf("Benchmarking memory access patterns (%d elements):\n\n", N);

  // Benchmark coalesced
  benchmark(
      "Coalesced   ",
      [](const float *in, float *out, int n) {
        coalesced_read<<<n / 256, 256>>>(in, out, n);
      },
      d_in, d_out, N, 100);

  // Benchmark strided (stride=32 is worst case for warp)
  printf("\n");
  benchmark(
      "Strided (32)",
      [](const float *in, float *out, int n) {
        strided_read<<<n / 256, 256>>>(in, out, n / 32, 32);
      },
      d_in, d_out, N, 100);

  cudaFree(d_in);
  cudaFree(d_out);
  delete[] h_data;

  return 0;
}
