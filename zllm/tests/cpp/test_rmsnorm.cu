// =============================================================================
// test_rmsnorm.cu - RMSNorm Unit Tests
// =============================================================================

#include "../../csrc/include/common.cuh"
#include "../../csrc/kernels/rmsnorm.cu"
#include "../../csrc/kernels/rmsnorm.cuh"
#include <cmath>
#include <cstdio>
#include <cuda_runtime.h>
#include <random>
#include <vector>

using namespace mini_vllm;

// CPU reference implementation
void rmsnorm_cpu(float *output, const float *input, const float *weight,
                 int num_tokens, int hidden_dim, float epsilon) {
  for (int row = 0; row < num_tokens; row++) {
    const float *row_in = input + row * hidden_dim;
    float *row_out = output + row * hidden_dim;

    // Compute sum of squares
    float sum_sq = 0.0f;
    for (int i = 0; i < hidden_dim; i++) {
      sum_sq += row_in[i] * row_in[i];
    }

    // Compute RMS
    float rms = sqrtf(sum_sq / hidden_dim + epsilon);
    float rrms = 1.0f / rms;

    // Normalize
    for (int i = 0; i < hidden_dim; i++) {
      row_out[i] = row_in[i] * rrms * weight[i];
    }
  }
}

// Calculate max absolute error
float max_abs_error(const float *a, const float *b, int n) {
  float max_err = 0.0f;
  for (int i = 0; i < n; i++) {
    float err = fabsf(a[i] - b[i]);
    if (err > max_err)
      max_err = err;
  }
  return max_err;
}

int main() {
  printf("=== RMSNorm Kernel Tests ===\n\n");

  // Test configuration
  const int num_tokens = 128;
  const int hidden_dim = 4096;
  const float epsilon = 1e-6f;

  const size_t size = num_tokens * hidden_dim * sizeof(float);
  const size_t weight_size = hidden_dim * sizeof(float);

  // Allocate host memory
  std::vector<float> h_input(num_tokens * hidden_dim);
  std::vector<float> h_weight(hidden_dim);
  std::vector<float> h_output_cpu(num_tokens * hidden_dim);
  std::vector<float> h_output_gpu(num_tokens * hidden_dim);

  // Initialize with random values
  std::mt19937 gen(42);
  std::normal_distribution<float> dist(0.0f, 1.0f);

  for (auto &v : h_input)
    v = dist(gen);
  for (auto &v : h_weight)
    v = dist(gen) * 0.1f + 1.0f; // ~1.0

  // Allocate device memory
  float *d_input, *d_weight, *d_output;
  cudaMalloc(&d_input, size);
  cudaMalloc(&d_weight, weight_size);
  cudaMalloc(&d_output, size);

  // Copy to device
  cudaMemcpy(d_input, h_input.data(), size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_weight, h_weight.data(), weight_size, cudaMemcpyHostToDevice);

  // =======================================================================
  // Test 1: Correctness
  // =======================================================================
  printf("Test 1: Correctness\n");

  // CPU reference
  rmsnorm_cpu(h_output_cpu.data(), h_input.data(), h_weight.data(), num_tokens,
              hidden_dim, epsilon);

  // GPU
  rmsnorm_forward_optimized(d_output, d_input, d_weight, num_tokens, hidden_dim,
                            epsilon, nullptr);
  cudaDeviceSynchronize();

  // Copy back
  cudaMemcpy(h_output_gpu.data(), d_output, size, cudaMemcpyDeviceToHost);

  // Compare
  float max_err = max_abs_error(h_output_cpu.data(), h_output_gpu.data(),
                                num_tokens * hidden_dim);

  printf("  Max absolute error: %.2e\n", max_err);
  if (max_err < 1e-5f) {
    printf("  [PASS] Results match within tolerance\n");
  } else {
    printf("  [FAIL] Error too large!\n");
  }

  // =======================================================================
  // Test 2: Performance
  // =======================================================================
  printf("\nTest 2: Performance\n");

  // Warmup
  for (int i = 0; i < 10; i++) {
    rmsnorm_forward_optimized(d_output, d_input, d_weight, num_tokens,
                              hidden_dim, epsilon, nullptr);
  }
  cudaDeviceSynchronize();

  // Benchmark
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  const int iterations = 1000;

  cudaEventRecord(start);
  for (int i = 0; i < iterations; i++) {
    rmsnorm_forward_optimized(d_output, d_input, d_weight, num_tokens,
                              hidden_dim, epsilon, nullptr);
  }
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);

  float ms;
  cudaEventElapsedTime(&ms, start, stop);
  float avg_ms = ms / iterations;

  // Calculate bandwidth
  // Read: input (N*H) + weight (H)
  // Write: output (N*H)
  float bytes = (2.0f * num_tokens * hidden_dim + hidden_dim) * sizeof(float) *
                iterations;
  float bandwidth_gb = bytes / (ms * 1e6f);

  printf("  Time per call: %.3f us\n", avg_ms * 1000);
  printf("  Bandwidth: %.1f GB/s\n", bandwidth_gb);
  printf("  Tokens/sec: %.1f M\n", (num_tokens * iterations) / (ms * 1e3f));

  // =======================================================================
  // Test 3: Edge cases
  // =======================================================================
  printf("\nTest 3: Edge cases\n");

  // Test with different hidden dimensions
  int test_dims[] = {256, 512, 1024, 2048, 4096, 8192};
  for (int dim : test_dims) {
    std::vector<float> small_input(num_tokens * dim);
    std::vector<float> small_weight(dim);
    std::vector<float> small_cpu(num_tokens * dim);
    std::vector<float> small_gpu(num_tokens * dim);

    for (float &v : small_input)
      v = dist(gen);
    for (float &v : small_weight)
      v = 1.0f;

    float *d_si, *d_sw, *d_so;
    cudaMalloc(&d_si, num_tokens * dim * sizeof(float));
    cudaMalloc(&d_sw, dim * sizeof(float));
    cudaMalloc(&d_so, num_tokens * dim * sizeof(float));

    cudaMemcpy(d_si, small_input.data(), num_tokens * dim * sizeof(float),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_sw, small_weight.data(), dim * sizeof(float),
               cudaMemcpyHostToDevice);

    rmsnorm_cpu(small_cpu.data(), small_input.data(), small_weight.data(),
                num_tokens, dim, epsilon);
    rmsnorm_forward_optimized(d_so, d_si, d_sw, num_tokens, dim, epsilon,
                              nullptr);
    cudaDeviceSynchronize();

    cudaMemcpy(small_gpu.data(), d_so, num_tokens * dim * sizeof(float),
               cudaMemcpyDeviceToHost);

    float err =
        max_abs_error(small_cpu.data(), small_gpu.data(), num_tokens * dim);
    printf("  hidden_dim=%d: max_err=%.2e %s\n", dim, err,
           err < 1e-5f ? "[PASS]" : "[FAIL]");

    cudaFree(d_si);
    cudaFree(d_sw);
    cudaFree(d_so);
  }

  // Cleanup
  cudaFree(d_input);
  cudaFree(d_weight);
  cudaFree(d_output);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  printf("\n=== Tests Complete ===\n");
  return 0;
}
