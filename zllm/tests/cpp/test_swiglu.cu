// =============================================================================
// test_swiglu.cu - SwiGLU Unit Tests
// =============================================================================

#include "../../csrc/kernels/swiglu.cuh"
#include <cmath>
#include <cstdio>
#include <cuda_runtime.h>
#include <random>
#include <vector>

using namespace mini_vllm;

// CPU reference implementation
float silu_cpu(float x) { return x / (1.0f + expf(-x)); }

void swiglu_cpu(float *output, const float *gate, const float *up, int size) {
  for (int i = 0; i < size; i++) {
    output[i] = silu_cpu(gate[i]) * up[i];
  }
}

float max_abs_error(const float *a, const float *b, int n) {
  float max_err = 0.0f;
  for (int i = 0; i < n; i++) {
    float err = fabsf(a[i] - b[i]);
    if (err > max_err)
      max_err = err;
  }
  return max_err;
}

float max_rel_error(const float *a, const float *b, int n) {
  float max_err = 0.0f;
  for (int i = 0; i < n; i++) {
    float err = fabsf(a[i] - b[i]) / (fabsf(a[i]) + 1e-6f);
    if (err > max_err)
      max_err = err;
  }
  return max_err;
}

int main() {
  printf("=== SwiGLU Kernel Tests ===\n\n");

  // Configuration (Qwen3-like)
  const int num_tokens = 128;
  const int intermediate_dim = 11008;
  const int total_size = num_tokens * intermediate_dim;

  // Allocate host memory
  std::vector<float> h_gate(total_size);
  std::vector<float> h_up(total_size);
  std::vector<float> h_output_cpu(total_size);
  std::vector<float> h_output_gpu(total_size);

  // Initialize with random values
  std::mt19937 gen(42);
  std::normal_distribution<float> dist(0.0f, 1.0f);

  for (auto &v : h_gate)
    v = dist(gen);
  for (auto &v : h_up)
    v = dist(gen);

  // Allocate device memory
  float *d_gate, *d_up, *d_output;
  cudaMalloc(&d_gate, total_size * sizeof(float));
  cudaMalloc(&d_up, total_size * sizeof(float));
  cudaMalloc(&d_output, total_size * sizeof(float));

  // Copy to device
  cudaMemcpy(d_gate, h_gate.data(), total_size * sizeof(float),
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_up, h_up.data(), total_size * sizeof(float),
             cudaMemcpyHostToDevice);

  // =======================================================================
  // Test 1: Correctness
  // =======================================================================
  printf("Test 1: Correctness\n");

  // CPU reference
  swiglu_cpu(h_output_cpu.data(), h_gate.data(), h_up.data(), total_size);

  // GPU
  swiglu_forward(d_output, d_gate, d_up, num_tokens, intermediate_dim);
  cudaDeviceSynchronize();

  // Copy back
  cudaMemcpy(h_output_gpu.data(), d_output, total_size * sizeof(float),
             cudaMemcpyDeviceToHost);

  // Compare
  float abs_err =
      max_abs_error(h_output_cpu.data(), h_output_gpu.data(), total_size);
  float rel_err =
      max_rel_error(h_output_cpu.data(), h_output_gpu.data(), total_size);

  printf("  Max absolute error: %.2e\n", abs_err);
  printf("  Max relative error: %.2e\n", rel_err);
  if (abs_err < 1e-5f) {
    printf("  [PASS] Results match\n");
  } else {
    printf("  [FAIL] Error too large!\n");
  }

  // =======================================================================
  // Test 2: Swish function properties
  // =======================================================================
  printf("\nTest 2: Swish function properties\n");

  // Swish(0) should be 0
  float test_val = silu_cpu(0.0f);
  printf("  Swish(0) = %.6f (expected: 0)\n", test_val);

  // Swish(x) ≈ x for large positive x
  test_val = silu_cpu(100.0f);
  printf("  Swish(100) = %.6f (expected: ~100)\n", test_val);

  // Swish(x) ≈ 0 for large negative x
  test_val = silu_cpu(-100.0f);
  printf("  Swish(-100) = %.2e (expected: ~0)\n", fabsf(test_val));

  // Swish is smooth (derivative exists everywhere)
  // Swish'(x) = sigmoid(x) + x * sigmoid(x) * (1 - sigmoid(x))
  //           = sigmoid(x) * (1 + x * (1 - sigmoid(x)))
  printf("  [INFO] Swish is differentiable everywhere\n");

  // =======================================================================
  // Test 3: Performance
  // =======================================================================
  printf("\nTest 3: Performance\n");

  // Warmup
  for (int i = 0; i < 10; i++) {
    swiglu_forward(d_output, d_gate, d_up, num_tokens, intermediate_dim);
  }
  cudaDeviceSynchronize();

  // Benchmark
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  const int iterations = 1000;
  cudaEventRecord(start);
  for (int i = 0; i < iterations; i++) {
    swiglu_forward(d_output, d_gate, d_up, num_tokens, intermediate_dim);
  }
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);

  float ms;
  cudaEventElapsedTime(&ms, start, stop);
  float avg_us = (ms / iterations) * 1000;

  // Calculate bandwidth
  // Read: gate + up (2 * size * 4 bytes)
  // Write: output (1 * size * 4 bytes)
  float bytes = 3.0f * total_size * sizeof(float) * iterations;
  float bandwidth_gb = bytes / (ms * 1e6f);

  // Calculate operations
  // For each element: sigmoid (exp, add, div) + mul + mul
  // Roughly 5 FLOPs per element
  float flops = 5.0f * total_size * iterations;
  float gflops = flops / (ms * 1e6f);

  printf("  Time per call: %.2f us\n", avg_us);
  printf("  Bandwidth: %.1f GB/s\n", bandwidth_gb);
  printf("  Compute: %.1f GFLOP/s\n", gflops);
  printf("  Throughput: %.1f M elements/sec\n",
         (total_size * iterations) / (ms * 1e3f));

  // =======================================================================
  // Test 4: Different sizes
  // =======================================================================
  printf("\nTest 4: Different intermediate dimensions\n");

  int test_dims[] = {1024, 4096, 8192, 11008, 16384};

  for (int dim : test_dims) {
    int size = num_tokens * dim;

    std::vector<float> g(size), u(size), cpu_res(size), gpu_res(size);
    for (auto &v : g)
      v = dist(gen);
    for (auto &v : u)
      v = dist(gen);

    float *dg, *du, *dout;
    cudaMalloc(&dg, size * sizeof(float));
    cudaMalloc(&du, size * sizeof(float));
    cudaMalloc(&dout, size * sizeof(float));

    cudaMemcpy(dg, g.data(), size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(du, u.data(), size * sizeof(float), cudaMemcpyHostToDevice);

    swiglu_cpu(cpu_res.data(), g.data(), u.data(), size);
    swiglu_forward(dout, dg, du, num_tokens, dim);
    cudaDeviceSynchronize();

    cudaMemcpy(gpu_res.data(), dout, size * sizeof(float),
               cudaMemcpyDeviceToHost);

    float err = max_abs_error(cpu_res.data(), gpu_res.data(), size);
    printf("  dim=%5d: max_err=%.2e %s\n", dim, err,
           err < 1e-5f ? "[PASS]" : "[FAIL]");

    cudaFree(dg);
    cudaFree(du);
    cudaFree(dout);
  }

  // Cleanup
  cudaFree(d_gate);
  cudaFree(d_up);
  cudaFree(d_output);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  printf("\n=== Tests Complete ===\n");
  return 0;
}
