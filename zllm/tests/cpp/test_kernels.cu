// =============================================================================
// test_cuda_setup.cu - Verify CUDA is working
// =============================================================================

#define __CUDA_NO_HALF_OPERATORS__
#define __CUDA_NO_HALF_CONVERSIONS__
#define __CUDA_NO_BFLOAT16_CONVERSIONS__

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

__global__ void hello_kernel() {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    printf("Hello from CUDA kernel!\n");
  }
}

int main() {
  // Get device info
  int device;
  cudaGetDevice(&device);

  cudaDeviceProp props;
  cudaGetDeviceProperties(&props, device);

    printf("=== CUDA Device Info ===\n");
    printf("Device: %s\n", props.name);
    printf("Compute Capability: %d.%d\n", props.major, props.minor);
    printf("Total Memory: %.2f GB\n", props.totalGlobalMem / 1e9);
    printf("SMs: %d\n", props.multiProcessorCount);
    printf("Max Threads/Block: %d\n", props.maxThreadsPerBlock);
    printf("Shared Memory/Block: %.2f KB\n", props.sharedMemPerBlock / 1024.0);
    printf("\n");

  // Launch test kernel
  hello_kernel<<<1, 32>>>();
  cudaDeviceSynchronize();

  // Check for errors
  cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
      printf("CUDA Error: %s\n", cudaGetErrorString(err));
      return 1;
    }

    printf("CUDA setup verified successfully!\n");
  return 0;
}
