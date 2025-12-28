// =============================================================================
// test_rope.cu - RoPE Unit Tests
// =============================================================================

#include "../../csrc/kernels/rope.cuh"
#include <cmath>
#include <cstdio>
#include <cuda_runtime.h>
#include <random>
#include <vector>

using namespace mini_vllm;

// CPU reference implementation
void rope_cpu(float *query, float *key, const int *positions, int num_tokens,
              int num_heads, int num_kv_heads, int head_dim, float theta_base) {
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
  printf("=== RoPE Kernel Tests ===\n\n");

  // Configuration (Qwen3-like)
  const int num_tokens = 128;
  const int num_heads = 32;
  const int num_kv_heads = 8;
  const int head_dim = 128;
  const int max_seq_len = 8192;
  const float theta_base = 1000000.0f; // Qwen3 uses 1M

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

  for (auto &v : h_q_cpu)
    v = dist(gen);
  for (auto &v : h_k_cpu)
    v = dist(gen);
  h_q_gpu = h_q_cpu; // Copy for GPU
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
  cudaMemcpy(d_q, h_q_gpu.data(), q_size * sizeof(float),
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_k, h_k_gpu.data(), k_size * sizeof(float),
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_positions, h_positions.data(), num_tokens * sizeof(int),
             cudaMemcpyHostToDevice);

  // =======================================================================
  // Test 1: Correctness
  // =======================================================================
  printf("Test 1: Correctness\n");

  // CPU reference (modifies in-place)
  rope_cpu(h_q_cpu.data(), h_k_cpu.data(), h_positions.data(), num_tokens,
           num_heads, num_kv_heads, head_dim, theta_base);

  // GPU (modifies in-place)
  rope_forward(d_q, d_k, d_cos, d_sin, d_positions, num_tokens, num_heads,
               num_kv_heads, head_dim);
  cudaDeviceSynchronize();

  // Copy back
  cudaMemcpy(h_q_gpu.data(), d_q, q_size * sizeof(float),
             cudaMemcpyDeviceToHost);
  cudaMemcpy(h_k_gpu.data(), d_k, k_size * sizeof(float),
             cudaMemcpyDeviceToHost);

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

  for (auto &v : q1)
    v = dist(gen);
  for (auto &v : k2)
    v = dist(gen);
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
  for (auto &v : h_q_gpu)
    v = dist(gen);
  for (auto &v : h_k_gpu)
    v = dist(gen);
  cudaMemcpy(d_q, h_q_gpu.data(), q_size * sizeof(float),
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_k, h_k_gpu.data(), k_size * sizeof(float),
             cudaMemcpyHostToDevice);

  // Warmup
  for (int i = 0; i < 10; i++) {
    rope_forward(d_q, d_k, d_cos, d_sin, d_positions, num_tokens, num_heads,
                 num_kv_heads, head_dim);
  }
  cudaDeviceSynchronize();

  // Benchmark
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  const int iterations = 1000;
  cudaEventRecord(start);
  for (int i = 0; i < iterations; i++) {
    rope_forward(d_q, d_k, d_cos, d_sin, d_positions, num_tokens, num_heads,
                 num_kv_heads, head_dim);
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
