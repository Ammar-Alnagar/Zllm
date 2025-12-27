# Phase 0: Project Setup

## Table of Contents

1. [Directory Structure](#directory-structure)
2. [Creating the Project](#creating-the-project)
3. [CMake Configuration](#cmake-configuration)
4. [Python Package Setup](#python-package-setup)
5. [Environment Configuration](#environment-configuration)
6. [Verification](#verification)

---

## Directory Structure

Our mini-vLLM project follows this structure. Create each directory and file as we go through this guide.

```
mini_vllm/
├── CMakeLists.txt                 # Root CMake configuration
├── README.md                      # Project documentation
├── setup.py                       # Python package setup
├── pyproject.toml                 # Python build configuration
│
├── csrc/                          # C++/CUDA source code
│   ├── CMakeLists.txt             # C++ build configuration
│   │
│   ├── include/                   # Header files
│   │   ├── common.cuh             # Common CUDA utilities
│   │   ├── cuda_utils.cuh         # CUDA helper functions
│   │   └── types.hpp              # Type definitions
│   │
│   ├── kernels/                   # CUDA kernels
│   │   ├── CMakeLists.txt
│   │   ├── rmsnorm.cu             # RMSNorm kernel
│   │   ├── rmsnorm.cuh
│   │   ├── rope.cu                # RoPE kernel
│   │   ├── rope.cuh
│   │   ├── swiglu.cu              # SwiGLU kernel
│   │   └── swiglu.cuh
│   │
│   ├── attention/                 # Attention implementations
│   │   ├── CMakeLists.txt
│   │   ├── flash_attention.cu     # Flash Attention prefill
│   │   ├── flash_attention.cuh
│   │   ├── flash_infer.cu         # FlashInfer decode
│   │   ├── flash_infer.cuh
│   │   ├── paged_attention.cu     # Paged attention
│   │   └── paged_attention.cuh
│   │
│   ├── memory/                    # Memory management
│   │   ├── CMakeLists.txt
│   │   ├── block_manager.cpp      # Block allocation
│   │   ├── block_manager.hpp
│   │   ├── kv_cache.cu            # KV cache operations
│   │   ├── kv_cache.cuh
│   │   ├── memory_pool.cpp        # Memory pool
│   │   └── memory_pool.hpp
│   │
│   └── bindings/                  # Python bindings
│       ├── CMakeLists.txt
│       └── bindings.cpp           # pybind11 bindings
│
├── python/                        # Python package
│   └── mini_vllm/
│       ├── __init__.py
│       ├── config.py              # Model configuration
│       ├── model.py               # Model implementation
│       ├── tokenizer.py           # Tokenizer wrapper
│       ├── sampling.py            # Sampling strategies
│       ├── scheduler.py           # Request scheduler
│       ├── radix_tree.py          # Radix tree for prefix sharing
│       ├── engine.py              # Inference engine
│       └── server.py              # FastAPI server
│
├── tests/                         # Test files
│   ├── cpp/                       # C++ tests
│   │   ├── CMakeLists.txt
│   │   └── test_kernels.cpp
│   └── python/                    # Python tests
│       ├── test_model.py
│       ├── test_scheduler.py
│       └── test_radix_tree.py
│
└── benchmarks/                    # Performance benchmarks
    ├── benchmark_attention.py
    └── benchmark_inference.py
```

---

## Creating the Project

### Step 1: Create Root Directory

Open your terminal and run these commands:

```bash
# Navigate to your workspace
cd ~/work

# Create project directory
mkdir -p mini_vllm
cd mini_vllm

# Create directory structure
mkdir -p csrc/{include,kernels,attention,memory,bindings}
mkdir -p python/mini_vllm
mkdir -p tests/{cpp,python}
mkdir -p benchmarks

# Verify structure
find . -type d | head -20
```

### Step 2: Create Root CMakeLists.txt

Create file: `mini_vllm/CMakeLists.txt`

```cmake
# ==============================================================================
# Mini-vLLM: Root CMake Configuration
# ==============================================================================
# This is the main CMake file that configures the entire project build.
# It sets up CUDA, finds dependencies, and includes subdirectories.
# ==============================================================================

cmake_minimum_required(VERSION 3.24 FATAL_ERROR)

# ------------------------------------------------------------------------------
# Project Definition
# ------------------------------------------------------------------------------
project(mini_vllm
    VERSION 0.1.0
    DESCRIPTION "Mini vLLM - Educational LLM Inference Engine"
    LANGUAGES CXX CUDA
)

# ------------------------------------------------------------------------------
# C++ Standard Configuration
# ------------------------------------------------------------------------------
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ------------------------------------------------------------------------------
# CUDA Configuration
# ------------------------------------------------------------------------------
# Enable separable compilation for device code linking
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)

# Set CUDA architectures - adjust based on your GPU
# 80 = Ampere (A100, RTX 30xx)
# 86 = Ampere (RTX 30xx laptop)
# 89 = Ada Lovelace (RTX 40xx)
# 90 = Hopper (H100)
set(CMAKE_CUDA_ARCHITECTURES "80;86;89;90")

# CUDA compilation flags
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-extended-lambda")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -lineinfo")

# Debug vs Release flags
set(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS_DEBUG} -G -g")
set(CMAKE_CUDA_FLAGS_RELEASE "${CMAKE_CUDA_FLAGS_RELEASE} -O3")

# ------------------------------------------------------------------------------
# Build Type
# ------------------------------------------------------------------------------
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")

# ------------------------------------------------------------------------------
# Find Dependencies
# ------------------------------------------------------------------------------

# CUDA Toolkit (provides cuBLAS, etc.)
find_package(CUDAToolkit REQUIRED)
message(STATUS "Found CUDA ${CUDAToolkit_VERSION}")

# Python for pybind11
find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
message(STATUS "Found Python ${Python3_VERSION}")

# pybind11 for Python bindings
find_package(pybind11 CONFIG REQUIRED)
message(STATUS "Found pybind11")

# ------------------------------------------------------------------------------
# Compiler Warnings
# ------------------------------------------------------------------------------
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# ------------------------------------------------------------------------------
# Include Directories
# ------------------------------------------------------------------------------
include_directories(${CMAKE_SOURCE_DIR}/csrc/include)
include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

# ------------------------------------------------------------------------------
# Add Subdirectories
# ------------------------------------------------------------------------------
add_subdirectory(csrc)

# Enable testing
enable_testing()
add_subdirectory(tests/cpp)

# ------------------------------------------------------------------------------
# Installation
# ------------------------------------------------------------------------------
install(TARGETS mini_vllm_ops
    LIBRARY DESTINATION ${Python3_SITELIB}/mini_vllm
)

# ------------------------------------------------------------------------------
# Summary
# ------------------------------------------------------------------------------
message(STATUS "")
message(STATUS "=== Mini-vLLM Configuration Summary ===")
message(STATUS "Version:          ${PROJECT_VERSION}")
message(STATUS "Build type:       ${CMAKE_BUILD_TYPE}")
message(STATUS "CUDA version:     ${CUDAToolkit_VERSION}")
message(STATUS "Python version:   ${Python3_VERSION}")
message(STATUS "Install prefix:   ${CMAKE_INSTALL_PREFIX}")
message(STATUS "========================================")
message(STATUS "")
```

### Step 3: Create csrc/CMakeLists.txt

Create file: `mini_vllm/csrc/CMakeLists.txt`

```cmake
# ==============================================================================
# Mini-vLLM: C++/CUDA Source CMake Configuration
# ==============================================================================

# ------------------------------------------------------------------------------
# Collect Source Files
# ------------------------------------------------------------------------------

# Kernel sources
set(KERNEL_SOURCES
    kernels/rmsnorm.cu
    kernels/rope.cu
    kernels/swiglu.cu
)

# Attention sources
set(ATTENTION_SOURCES
    attention/flash_attention.cu
    attention/flash_infer.cu
    attention/paged_attention.cu
)

# Memory management sources
set(MEMORY_SOURCES
    memory/block_manager.cpp
    memory/kv_cache.cu
    memory/memory_pool.cpp
)

# Python bindings
set(BINDING_SOURCES
    bindings/bindings.cpp
)

# ------------------------------------------------------------------------------
# Create the Python Extension Module
# ------------------------------------------------------------------------------
pybind11_add_module(mini_vllm_ops
    ${KERNEL_SOURCES}
    ${ATTENTION_SOURCES}
    ${MEMORY_SOURCES}
    ${BINDING_SOURCES}
)

# ------------------------------------------------------------------------------
# Include Directories
# ------------------------------------------------------------------------------
target_include_directories(mini_vllm_ops PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/kernels
    ${CMAKE_CURRENT_SOURCE_DIR}/attention
    ${CMAKE_CURRENT_SOURCE_DIR}/memory
)

# ------------------------------------------------------------------------------
# Link Libraries
# ------------------------------------------------------------------------------
target_link_libraries(mini_vllm_ops PRIVATE
    CUDA::cudart
    CUDA::cublas
    CUDA::curand
)

# ------------------------------------------------------------------------------
# CUDA Properties
# ------------------------------------------------------------------------------
set_target_properties(mini_vllm_ops PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
)

# ------------------------------------------------------------------------------
# Create Static Library for Testing
# ------------------------------------------------------------------------------
add_library(mini_vllm_static STATIC
    ${KERNEL_SOURCES}
    ${ATTENTION_SOURCES}
    ${MEMORY_SOURCES}
)

target_include_directories(mini_vllm_static PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/kernels
    ${CMAKE_CURRENT_SOURCE_DIR}/attention
    ${CMAKE_CURRENT_SOURCE_DIR}/memory
)

target_link_libraries(mini_vllm_static PUBLIC
    CUDA::cudart
    CUDA::cublas
    CUDA::curand
)

set_target_properties(mini_vllm_static PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
)
```

---

## Core Header Files

### Step 4: Create Common CUDA Header

Create file: `mini_vllm/csrc/include/common.cuh`

```c++
// =============================================================================
// common.cuh - Common CUDA Utilities
// =============================================================================
// This header provides common utilities used across all CUDA kernels:
// - Error checking macros
// - Warp-level primitives
// - Memory alignment helpers
// - Common constants
// =============================================================================

#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cstdio>
#include <cstdlib>

// =============================================================================
// CUDA Error Checking
// =============================================================================

/**
 * CUDA_CHECK - Macro for checking CUDA API call results
 *
 * Usage:
 *     CUDA_CHECK(cudaMalloc(&ptr, size));
 *
 * If the call fails, prints error message with file/line and exits
 */
#define CUDA_CHECK(call)                                                      \
    do {                                                                       \
        cudaError_t err = call;                                               \
        if (err != cudaSuccess) {                                             \
            fprintf(stderr, "CUDA Error at %s:%d - %s\n",                     \
                    __FILE__, __LINE__, cudaGetErrorString(err));             \
            exit(EXIT_FAILURE);                                               \
        }                                                                      \
    } while (0)

/**
 * CUDA_CHECK_LAST - Check for errors after kernel launch
 *
 * Usage:
 *     my_kernel<<<grid, block>>>(...);
 *     CUDA_CHECK_LAST();
 */
#define CUDA_CHECK_LAST()                                                     \
    do {                                                                       \
        cudaError_t err = cudaGetLastError();                                 \
        if (err != cudaSuccess) {                                             \
            fprintf(stderr, "CUDA Kernel Error at %s:%d - %s\n",              \
                    __FILE__, __LINE__, cudaGetErrorString(err));             \
            exit(EXIT_FAILURE);                                               \
        }                                                                      \
    } while (0)

// =============================================================================
// Constants
// =============================================================================

// Warp size is fixed at 32 threads for all NVIDIA GPUs
constexpr int WARP_SIZE = 32;

// Maximum threads per block (hardware limit)
constexpr int MAX_THREADS_PER_BLOCK = 1024;

// Maximum shared memory per block (48KB typical, 164KB on Hopper)
constexpr int MAX_SHARED_MEMORY = 48 * 1024;

// Cache line size for memory coalescing (128 bytes)
constexpr int CACHE_LINE_SIZE = 128;

// KV cache block size (16 tokens per block)
constexpr int KV_BLOCK_SIZE = 16;

// =============================================================================
// Type Aliases
// =============================================================================

using fp16 = half;        // FP16 type
using bf16 = __nv_bfloat16;  // BF16 type

// =============================================================================
// Device Functions - Warp Primitives
// =============================================================================

/**
 * warp_reduce_sum - Sum reduction within a warp
 *
 * Uses shuffle instructions for efficient intra-warp communication.
 * All threads in the warp must participate.
 *
 * @param val: Value to reduce from each thread
 * @return: Sum of all values in the warp (in lane 0, broadcast to all)
 */
template<typename T>
__device__ __forceinline__ T warp_reduce_sum(T val) {
    // Shuffle down and add, halving the distance each iteration
    // This creates a tree reduction pattern
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    // Broadcast result from lane 0 to all lanes
    return __shfl_sync(0xffffffff, val, 0);
}

/**
 * warp_reduce_max - Maximum reduction within a warp
 *
 * @param val: Value to reduce from each thread
 * @return: Maximum value in the warp
 */
template<typename T>
__device__ __forceinline__ T warp_reduce_max(T val) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val = max(val, __shfl_down_sync(0xffffffff, val, offset));
    }
    return __shfl_sync(0xffffffff, val, 0);
}

/**
 * block_reduce_sum - Sum reduction across an entire block
 *
 * Uses shared memory for inter-warp communication.
 *
 * @param val: Value to reduce from each thread
 * @param shared: Pointer to shared memory (size = num_warps)
 * @return: Sum of all values in the block (in thread 0, broadcast to all)
 */
template<typename T>
__device__ __forceinline__ T block_reduce_sum(T val, T* shared) {
    const int lane = threadIdx.x % WARP_SIZE;
    const int warp_id = threadIdx.x / WARP_SIZE;
    const int num_warps = blockDim.x / WARP_SIZE;

    // First, reduce within each warp
    val = warp_reduce_sum(val);

    // Write warp results to shared memory
    if (lane == 0) {
        shared[warp_id] = val;
    }
    __syncthreads();

    // First warp reduces all warp results
    if (warp_id == 0) {
        val = (lane < num_warps) ? shared[lane] : T(0);
        val = warp_reduce_sum(val);
    }
    __syncthreads();

    // Broadcast final result
    return __shfl_sync(0xffffffff, val, 0);
}

// =============================================================================
// Memory Helpers
// =============================================================================

/**
 * align_up - Align size up to nearest multiple of alignment
 *
 * @param size: Size to align
 * @param alignment: Alignment boundary (must be power of 2)
 * @return: Aligned size
 */
__host__ __device__ __forceinline__
size_t align_up(size_t size, size_t alignment) {
    return (size + alignment - 1) & ~(alignment - 1);
}

/**
 * div_ceil - Integer division, rounding up
 *
 * @param a: Dividend
 * @param b: Divisor
 * @return: Ceiling of a/b
 */
template<typename T>
__host__ __device__ __forceinline__
T div_ceil(T a, T b) {
    return (a + b - 1) / b;
}

// =============================================================================
// Numeric Helpers
// =============================================================================

/**
 * fast_rsqrt - Fast reciprocal square root (1/sqrt(x))
 *
 * Uses hardware rsqrt instruction for speed
 */
__device__ __forceinline__ float fast_rsqrt(float x) {
    return rsqrtf(x);
}

/**
 * silu - Sigmoid Linear Unit activation (Swish)
 *
 * silu(x) = x * sigmoid(x) = x / (1 + exp(-x))
 */
__device__ __forceinline__ float silu(float x) {
    return x / (1.0f + expf(-x));
}

// =============================================================================
// FP16/BF16 Conversion Helpers
// =============================================================================

__device__ __forceinline__ float fp16_to_float(fp16 x) {
    return __half2float(x);
}

__device__ __forceinline__ fp16 float_to_fp16(float x) {
    return __float2half(x);
}

__device__ __forceinline__ float bf16_to_float(bf16 x) {
    return __bfloat162float(x);
}

__device__ __forceinline__ bf16 float_to_bf16(float x) {
    return __float2bfloat16(x);
}

// =============================================================================
// Debug Helpers
// =============================================================================

#ifdef DEBUG_KERNELS
#define KERNEL_DEBUG_PRINT(fmt, ...) \
    if (threadIdx.x == 0 && blockIdx.x == 0) { \
        printf("[KERNEL DEBUG] " fmt "\n", ##__VA_ARGS__); \
    }
#else
#define KERNEL_DEBUG_PRINT(fmt, ...)
#endif
```

### Step 5: Create CUDA Utilities Header

Create file: `mini_vllm/csrc/include/cuda_utils.cuh`

```c++
// =============================================================================
// cuda_utils.cuh - CUDA Utility Functions
// =============================================================================
// Higher-level utilities for kernel configuration, memory management,
// and performance optimization.
// =============================================================================

#pragma once

#include "common.cuh"
#include <cuda_runtime.h>

namespace mini_vllm {

// =============================================================================
// Kernel Launch Configuration
// =============================================================================

/**
 * KernelConfig - Structure for kernel launch parameters
 */
struct KernelConfig {
    dim3 grid;
    dim3 block;
    size_t shared_memory;
    cudaStream_t stream;

    KernelConfig(dim3 g, dim3 b, size_t smem = 0, cudaStream_t s = 0)
        : grid(g), block(b), shared_memory(smem), stream(s) {}
};

/**
 * get_optimal_block_size - Calculate optimal block size for a kernel
 *
 * Uses occupancy API to find the best block size for maximum occupancy.
 *
 * @tparam Kernel: Kernel function type
 * @param kernel: Pointer to kernel function
 * @param shared_memory: Dynamic shared memory per block
 * @return: Optimal block size
 */
template<typename Kernel>
int get_optimal_block_size(Kernel kernel, size_t shared_memory = 0) {
    int min_grid_size;
    int block_size;

    CUDA_CHECK(cudaOccupancyMaxPotentialBlockSize(
        &min_grid_size, &block_size, kernel, shared_memory, 0
    ));

    return block_size;
}

/**
 * get_num_sms - Get number of streaming multiprocessors on current device
 */
inline int get_num_sms() {
    int device;
    CUDA_CHECK(cudaGetDevice(&device));

    int num_sms;
    CUDA_CHECK(cudaDeviceGetAttribute(
        &num_sms, cudaDevAttrMultiProcessorCount, device
    ));

    return num_sms;
}

/**
 * get_max_shared_memory - Get maximum shared memory per block
 */
inline size_t get_max_shared_memory() {
    int device;
    CUDA_CHECK(cudaGetDevice(&device));

    int max_shared;
    CUDA_CHECK(cudaDeviceGetAttribute(
        &max_shared, cudaDevAttrMaxSharedMemoryPerBlock, device
    ));

    return static_cast<size_t>(max_shared);
}

// =============================================================================
// Stream Management
// =============================================================================

/**
 * StreamPool - Pool of CUDA streams for concurrent operations
 */
class StreamPool {
public:
    StreamPool(int num_streams = 4) : num_streams_(num_streams) {
        streams_.resize(num_streams);
        for (int i = 0; i < num_streams; i++) {
            CUDA_CHECK(cudaStreamCreate(&streams_[i]));
        }
    }

    ~StreamPool() {
        for (auto& stream : streams_) {
            cudaStreamDestroy(stream);
        }
    }

    cudaStream_t get(int index) {
        return streams_[index % num_streams_];
    }

    void synchronize_all() {
        for (auto& stream : streams_) {
            CUDA_CHECK(cudaStreamSynchronize(stream));
        }
    }

private:
    int num_streams_;
    std::vector<cudaStream_t> streams_;
};

// =============================================================================
// GPU Memory Management
// =============================================================================

/**
 * GPUAllocator - Simple GPU memory allocator with tracking
 */
class GPUAllocator {
public:
    GPUAllocator() : total_allocated_(0), peak_allocated_(0) {}

    void* allocate(size_t size) {
        void* ptr;
        CUDA_CHECK(cudaMalloc(&ptr, size));

        total_allocated_ += size;
        peak_allocated_ = std::max(peak_allocated_, total_allocated_);
        allocations_[ptr] = size;

        return ptr;
    }

    void deallocate(void* ptr) {
        auto it = allocations_.find(ptr);
        if (it != allocations_.end()) {
            total_allocated_ -= it->second;
            allocations_.erase(it);
        }
        CUDA_CHECK(cudaFree(ptr));
    }

    size_t get_allocated() const { return total_allocated_; }
    size_t get_peak() const { return peak_allocated_; }

    void print_stats() const {
        printf("GPU Memory: %.2f MB allocated, %.2f MB peak\n",
               total_allocated_ / (1024.0 * 1024.0),
               peak_allocated_ / (1024.0 * 1024.0));
    }

private:
    size_t total_allocated_;
    size_t peak_allocated_;
    std::unordered_map<void*, size_t> allocations_;
};

// =============================================================================
// RAII Wrappers
// =============================================================================

/**
 * GPUBuffer - RAII wrapper for GPU memory
 */
template<typename T>
class GPUBuffer {
public:
    GPUBuffer() : ptr_(nullptr), size_(0) {}

    explicit GPUBuffer(size_t count) : size_(count) {
        CUDA_CHECK(cudaMalloc(&ptr_, count * sizeof(T)));
    }

    ~GPUBuffer() {
        if (ptr_) {
            cudaFree(ptr_);
        }
    }

    // Move semantics
    GPUBuffer(GPUBuffer&& other) noexcept
        : ptr_(other.ptr_), size_(other.size_) {
        other.ptr_ = nullptr;
        other.size_ = 0;
    }

    GPUBuffer& operator=(GPUBuffer&& other) noexcept {
        if (this != &other) {
            if (ptr_) cudaFree(ptr_);
            ptr_ = other.ptr_;
            size_ = other.size_;
            other.ptr_ = nullptr;
            other.size_ = 0;
        }
        return *this;
    }

    // Disable copy
    GPUBuffer(const GPUBuffer&) = delete;
    GPUBuffer& operator=(const GPUBuffer&) = delete;

    // Accessors
    T* data() { return ptr_; }
    const T* data() const { return ptr_; }
    size_t size() const { return size_; }
    size_t bytes() const { return size_ * sizeof(T); }

    // Copy from host
    void copy_from_host(const T* host_data, size_t count) {
        CUDA_CHECK(cudaMemcpy(ptr_, host_data, count * sizeof(T),
                              cudaMemcpyHostToDevice));
    }

    // Copy to host
    void copy_to_host(T* host_data, size_t count) const {
        CUDA_CHECK(cudaMemcpy(host_data, ptr_, count * sizeof(T),
                              cudaMemcpyDeviceToHost));
    }

    // Fill with value
    void fill(T value) {
        // Note: Simple implementation, use cudaMemset for zero
        std::vector<T> host_data(size_, value);
        copy_from_host(host_data.data(), size_);
    }

private:
    T* ptr_;
    size_t size_;
};

// =============================================================================
// Timing Utilities
// =============================================================================

/**
 * GPUTimer - Measure GPU kernel execution time using CUDA events
 */
class GPUTimer {
public:
    GPUTimer() {
        CUDA_CHECK(cudaEventCreate(&start_));
        CUDA_CHECK(cudaEventCreate(&stop_));
    }

    ~GPUTimer() {
        cudaEventDestroy(start_);
        cudaEventDestroy(stop_);
    }

    void start(cudaStream_t stream = 0) {
        CUDA_CHECK(cudaEventRecord(start_, stream));
    }

    void stop(cudaStream_t stream = 0) {
        CUDA_CHECK(cudaEventRecord(stop_, stream));
    }

    // Returns elapsed time in milliseconds
    float elapsed_ms() {
        CUDA_CHECK(cudaEventSynchronize(stop_));
        float ms;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start_, stop_));
        return ms;
    }

private:
    cudaEvent_t start_, stop_;
};

} // namespace mini_vllm
```

### Step 6: Create Types Header

Create file: `mini_vllm/csrc/include/types.hpp`

```cpp
// =============================================================================
// types.hpp - Type Definitions
// =============================================================================
// Common type definitions and data structures used throughout the project.
// =============================================================================

#pragma once

#include <cstdint>
#include <vector>
#include <memory>
#include <string>

namespace mini_vllm {

// =============================================================================
// Basic Types
// =============================================================================

// Data types supported by the inference engine
enum class DataType {
    FP32,    // 32-bit floating point
    FP16,    // 16-bit floating point (half precision)
    BF16,    // Brain floating point (16-bit)
    INT8,    // 8-bit integer (for quantization)
    INT4     // 4-bit integer (for quantization)
};

// Get size in bytes for a data type
inline size_t dtype_size(DataType dtype) {
    switch (dtype) {
        case DataType::FP32: return 4;
        case DataType::FP16: return 2;
        case DataType::BF16: return 2;
        case DataType::INT8: return 1;
        case DataType::INT4: return 1;  // Packed as 2 values per byte
        default: return 0;
    }
}

// =============================================================================
// Model Configuration
// =============================================================================

/**
 * ModelConfig - Configuration for a transformer model
 *
 * This matches the Qwen3 architecture with GQA support.
 */
struct ModelConfig {
    // Model dimensions
    int hidden_size = 4096;           // Hidden dimension (d_model)
    int intermediate_size = 11008;     // FFN intermediate dimension
    int num_hidden_layers = 32;        // Number of transformer layers
    int num_attention_heads = 32;      // Number of query heads
    int num_key_value_heads = 8;       // Number of KV heads (GQA)
    int head_dim = 128;                // Dimension per head
    int vocab_size = 152064;           // Vocabulary size (~150K)
    int max_position_embeddings = 8192; // Maximum sequence length

    // RoPE configuration
    float rope_theta = 1000000.0f;     // RoPE base frequency

    // Normalization
    float rms_norm_eps = 1e-6f;        // RMSNorm epsilon

    // Data type
    DataType dtype = DataType::FP16;

    // Derived values (computed from above)
    int kv_head_ratio() const {
        return num_attention_heads / num_key_value_heads;
    }

    size_t kv_cache_size_per_token() const {
        // K + V for each layer and each KV head
        return 2 * num_hidden_layers * num_key_value_heads * head_dim *
               dtype_size(dtype);
    }
};

// =============================================================================
// KV Cache Types
// =============================================================================

// Block size for paged KV cache (tokens per block)
constexpr int KV_BLOCK_SIZE = 16;

/**
 * BlockTable - Maps sequence positions to physical block indices
 *
 * For a sequence with position p, the block index is:
 *     block_idx = block_table[p / KV_BLOCK_SIZE]
 * The offset within the block is:
 *     offset = p % KV_BLOCK_SIZE
 */
using BlockTable = std::vector<int>;

/**
 * KVCacheBlock - Represents one block in the paged KV cache
 */
struct KVCacheBlock {
    int block_id;              // Unique identifier
    int ref_count;             // Reference count for sharing
    bool is_free;              // Whether block is available
    int num_tokens;            // Number of tokens stored (0 to KV_BLOCK_SIZE)

    KVCacheBlock(int id)
        : block_id(id), ref_count(0), is_free(true), num_tokens(0) {}
};

// =============================================================================
// Sequence Types
// =============================================================================

/**
 * SequenceState - Current state of a sequence in the batch
 */
enum class SequenceState {
    WAITING,     // Waiting in queue
    RUNNING,     // Currently being processed
    FINISHED,    // Generation complete
    PREEMPTED    // Temporarily stopped for higher priority
};

/**
 * SequenceData - Metadata for a single sequence
 */
struct SequenceData {
    int64_t seq_id;                    // Unique sequence identifier
    std::vector<int> token_ids;         // All tokens (prompt + generated)
    int prompt_len;                      // Original prompt length
    int output_len;                      // Number of generated tokens
    int max_output_len;                  // Maximum tokens to generate
    SequenceState state;                 // Current state
    BlockTable block_table;              // Physical block indices

    SequenceData(int64_t id, const std::vector<int>& prompt, int max_len)
        : seq_id(id)
        , token_ids(prompt)
        , prompt_len(prompt.size())
        , output_len(0)
        , max_output_len(max_len)
        , state(SequenceState::WAITING) {}

    int current_len() const { return prompt_len + output_len; }
    bool is_finished() const { return state == SequenceState::FINISHED; }

    int num_blocks_needed() const {
        return (current_len() + KV_BLOCK_SIZE - 1) / KV_BLOCK_SIZE;
    }
};

// =============================================================================
// Attention Types
// =============================================================================

/**
 * AttentionMetadata - Metadata for attention computation
 *
 * This structure is passed to attention kernels to describe the
 * current batch layout and KV cache configuration.
 */
struct AttentionMetadata {
    // Batch information
    int batch_size;                      // Number of sequences
    int max_seq_len;                     // Maximum sequence length in batch
    bool is_prefill;                     // Prefill phase (true) or decode (false)

    // Sequence lengths (size = batch_size)
    std::vector<int> seq_lens;           // Current length of each sequence
    std::vector<int> context_lens;       // Context length (KV cache length)

    // Block tables (size = batch_size x max_blocks)
    std::vector<int> block_tables;       // Flattened block tables
    int max_num_blocks;                  // Maximum blocks per sequence

    // Prefill-specific
    std::vector<int> query_start_loc;    // Start position of each query
    int total_query_len;                 // Total query tokens

    // Decode-specific
    std::vector<int> slot_mapping;       // Direct slot indices for decode
};

// =============================================================================
// Sampling Types
// =============================================================================

/**
 * SamplingParams - Parameters for token sampling
 */
struct SamplingParams {
    float temperature = 1.0f;            // Temperature for softmax
    float top_p = 1.0f;                  // Top-p (nucleus) sampling threshold
    int top_k = -1;                       // Top-k sampling (-1 = disabled)
    float repetition_penalty = 1.0f;     // Penalty for repeated tokens
    int max_tokens = 256;                 // Maximum tokens to generate
    std::vector<int> stop_token_ids;     // Stop generation on these tokens

    bool use_top_k() const { return top_k > 0; }
    bool use_top_p() const { return top_p < 1.0f; }
};

// =============================================================================
// Request Types
// =============================================================================

/**
 * Request - Represents a user request for text generation
 */
struct Request {
    int64_t request_id;                  // Unique request identifier
    std::string prompt;                   // Input text
    SamplingParams sampling_params;       // Sampling configuration
    int64_t arrival_time;                 // Timestamp of request arrival

    Request(int64_t id, const std::string& p, const SamplingParams& sp)
        : request_id(id)
        , prompt(p)
        , sampling_params(sp)
        , arrival_time(0) {}
};

/**
 * Response - Generated response for a request
 */
struct Response {
    int64_t request_id;                  // Matching request ID
    std::string generated_text;           // Generated output
    int num_tokens;                       // Number of tokens generated
    float generation_time_ms;             // Total generation time
    bool is_complete;                     // Whether generation finished

    Response(int64_t id)
        : request_id(id)
        , num_tokens(0)
        , generation_time_ms(0)
        , is_complete(false) {}
};

} // namespace mini_vllm
```

---

## Python Package Setup

### Step 7: Create setup.py

Create file: `mini_vllm/setup.py`

```python
"""
Mini-vLLM: Setup Script
=======================

This script configures the Python package installation, including
building the C++/CUDA extension module.

Usage:
    pip install -e .           # Development install
    pip install .              # Standard install
    python setup.py build_ext  # Build extension only
"""

import os
import sys
import subprocess
from pathlib import Path

from setuptools import setup, find_packages
from setuptools.command.build_ext import build_ext


class CMakeBuildExt(build_ext):
    """
    Custom build extension that uses CMake to build C++/CUDA code.

    This allows us to use CMake for the complex CUDA compilation while
    still integrating with Python's packaging tools.
    """

    def build_extensions(self):
        # Check for CMake
        try:
            subprocess.check_output(['cmake', '--version'])
        except OSError:
            raise RuntimeError("CMake must be installed to build this package")

        # Create build directory
        build_dir = Path(self.build_temp).absolute()
        build_dir.mkdir(parents=True, exist_ok=True)

        # Source directory
        source_dir = Path(__file__).parent.absolute()

        # CMake configuration
        cmake_args = [
            f'-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={self.build_lib}/mini_vllm',
            f'-DPYTHON_EXECUTABLE={sys.executable}',
            '-DCMAKE_BUILD_TYPE=Release',
        ]

        # Build arguments
        build_args = [
            '--config', 'Release',
            '--parallel', str(os.cpu_count() or 1),
        ]

        # Run CMake configure
        print(f"[CMake] Configuring in {build_dir}")
        subprocess.check_call(
            ['cmake', str(source_dir)] + cmake_args,
            cwd=build_dir
        )

        # Run CMake build
        print(f"[CMake] Building...")
        subprocess.check_call(
            ['cmake', '--build', '.'] + build_args,
            cwd=build_dir
        )


# Read README for long description
readme_path = Path(__file__).parent / "README.md"
long_description = ""
if readme_path.exists():
    long_description = readme_path.read_text(encoding='utf-8')


setup(
    name="mini_vllm",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Mini vLLM - Educational LLM Inference Engine",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/mini_vllm",

    # Package configuration
    packages=find_packages(where="python"),
    package_dir={"": "python"},

    # Include CUDA extension
    ext_modules=[],  # Handled by CMakeBuildExt
    cmdclass={'build_ext': CMakeBuildExt},

    # Python version requirement
    python_requires=">=3.10",

    # Dependencies
    install_requires=[
        "torch>=2.1.0",
        "tiktoken>=0.5.1",
        "fastapi>=0.104.0",
        "uvicorn>=0.24.0",
        "numpy>=1.24.0",
        "pydantic>=2.0.0",
    ],

    # Development dependencies
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=23.0.0",
            "isort>=5.0.0",
            "mypy>=1.0.0",
        ],
    },

    # Entry points
    entry_points={
        "console_scripts": [
            "mini-vllm-server=mini_vllm.server:main",
        ],
    },

    # Classifiers
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
```

### Step 8: Create pyproject.toml

Create file: `mini_vllm/pyproject.toml`

```toml
# =============================================================================
# pyproject.toml - Python Project Configuration
# =============================================================================

[build-system]
requires = [
    "setuptools>=65.0",
    "wheel",
    "cmake>=3.24",
    "ninja",
    "pybind11>=2.11",
]
build-backend = "setuptools.build_meta"

[project]
name = "mini_vllm"
version = "0.1.0"
description = "Mini vLLM - Educational LLM Inference Engine"
readme = "README.md"
license = {text = "MIT"}
requires-python = ">=3.10"
authors = [
    {name = "Your Name", email = "your.email@example.com"},
]
keywords = ["llm", "inference", "cuda", "transformer", "vllm"]

dependencies = [
    "torch>=2.1.0",
    "tiktoken>=0.5.1",
    "fastapi>=0.104.0",
    "uvicorn>=0.24.0",
    "numpy>=1.24.0",
    "pydantic>=2.0.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.0.0",
    "black>=23.0.0",
    "isort>=5.0.0",
    "mypy>=1.0.0",
    "pytest-asyncio>=0.21.0",
]

[project.scripts]
mini-vllm-server = "mini_vllm.server:main"

[tool.setuptools.packages.find]
where = ["python"]

[tool.black]
line-length = 88
target-version = ['py310']

[tool.isort]
profile = "black"
line_length = 88

[tool.mypy]
python_version = "3.10"
warn_return_any = true
warn_unused_configs = true
```

### Step 9: Create Python Package **init**.py

Create file: `mini_vllm/python/mini_vllm/__init__.py`

```python
"""
Mini-vLLM: Educational LLM Inference Engine
============================================

A from-scratch implementation of vLLM concepts for learning purposes.

This package provides:
- Custom CUDA kernels for attention, normalization, and activations
- Paged KV cache with RadixAttention for prefix sharing
- Continuous batching scheduler
- FastAPI server for deployment

Example:
    >>> from mini_vllm import LLMEngine, SamplingParams
    >>> engine = LLMEngine("Qwen/Qwen2-7B")
    >>> params = SamplingParams(temperature=0.7, max_tokens=100)
    >>> output = engine.generate("Hello, world!", params)
    >>> print(output.text)
"""

__version__ = "0.1.0"
__author__ = "Your Name"

# Import main components
from .config import ModelConfig, SamplingParams
from .engine import LLMEngine
from .tokenizer import Tokenizer

# Try to import CUDA extensions
try:
    from . import mini_vllm_ops as _ops
    CUDA_AVAILABLE = True
except ImportError:
    CUDA_AVAILABLE = False
    import warnings
    warnings.warn(
        "CUDA extensions not found. Install with: pip install -e .",
        ImportWarning
    )

__all__ = [
    "LLMEngine",
    "ModelConfig",
    "SamplingParams",
    "Tokenizer",
    "CUDA_AVAILABLE",
]
```

---

## Environment Configuration

### Step 10: Create Environment Script

Create file: `mini_vllm/scripts/setup_env.sh`

```bash
#!/bin/bash
# =============================================================================
# setup_env.sh - Environment Setup Script
# =============================================================================
# Run this script to set up the development environment.
# Usage: source scripts/setup_env.sh
# =============================================================================

set -e  # Exit on error

echo "=== Mini-vLLM Environment Setup ==="

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Check Python version
PYTHON_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
echo "Python version: $PYTHON_VERSION"

if [[ "$PYTHON_VERSION" < "3.10" ]]; then
    echo "Error: Python 3.10+ required"
    exit 1
fi

# Create virtual environment if it doesn't exist
VENV_DIR="$PROJECT_DIR/.venv"
if [[ ! -d "$VENV_DIR" ]]; then
    echo "Creating virtual environment..."
    python3 -m venv "$VENV_DIR"
fi

# Activate virtual environment
source "$VENV_DIR/bin/activate"
echo "Virtual environment activated: $VIRTUAL_ENV"

# Upgrade pip
pip install --upgrade pip

# Install PyTorch with CUDA
echo "Installing PyTorch with CUDA support..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install other dependencies
echo "Installing dependencies..."
pip install tiktoken==0.5.1
pip install fastapi==0.104.1
pip install uvicorn==0.24.0
pip install pybind11==2.11.1
pip install numpy==1.24.0
pip install pytest==7.4.3
pip install ninja  # For faster builds

# Verify CUDA
echo ""
echo "=== Verification ==="
python3 -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
"

# Set environment variables
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

echo ""
echo "=== Environment Ready ==="
echo "To activate: source $VENV_DIR/bin/activate"
echo "To build: cd $PROJECT_DIR && pip install -e ."
```

### Step 11: Create Build Script

Create file: `mini_vllm/scripts/build.sh`

```bash
#!/bin/bash
# =============================================================================
# build.sh - Build Script
# =============================================================================
# Build the C++/CUDA extensions.
# Usage: ./scripts/build.sh [debug|release]
# =============================================================================

set -e

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Build type
BUILD_TYPE="${1:-Release}"
echo "Build type: $BUILD_TYPE"

# Create build directory
BUILD_DIR="$PROJECT_DIR/build"
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

# Configure with CMake
echo "=== Configuring ==="
cmake "$PROJECT_DIR" \
    -DCMAKE_BUILD_TYPE="$BUILD_TYPE" \
    -GNinja

# Build
echo "=== Building ==="
ninja -j$(nproc)

echo "=== Build Complete ==="

# Run tests
if [[ "$2" == "--test" ]]; then
    echo "=== Running Tests ==="
    ctest --output-on-failure
fi
```

---

## Verification

### Step 12: Verify Project Structure

Run these commands to verify your setup:

```bash
cd ~/work/mini_vllm

# Check directory structure
echo "=== Directory Structure ==="
find . -type f -name "*.cpp" -o -name "*.cu" -o -name "*.cuh" -o -name "*.hpp" -o -name "*.py" -o -name "CMakeLists.txt" | head -30

# Check CMake
echo ""
echo "=== CMake Version ==="
cmake --version

# Check CUDA
echo ""
echo "=== CUDA Compiler ==="
nvcc --version

# Check Python
echo ""
echo "=== Python Environment ==="
python3 -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
```

### Step 13: Simple CUDA Test

Create file: `mini_vllm/tests/cpp/test_cuda_setup.cu`

```c++
// =============================================================================
// test_cuda_setup.cu - Verify CUDA is working
// =============================================================================

#include <cuda_runtime.h>
#include <cstdio>

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
    printf("Total Memory: %.1f GB\n", props.totalGlobalMem / 1e9);
    printf("SMs: %d\n", props.multiProcessorCount);
    printf("Max Threads/Block: %d\n", props.maxThreadsPerBlock);
    printf("Shared Memory/Block: %.1f KB\n", props.sharedMemPerBlock / 1024.0);
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
```

Compile and run:

```bash
nvcc -ccbin g++-14 -arch=sm_89 -o test_cuda zllm/tests/cpp/test_kernels.cu -lstdc++ && ./test_cuda
```

Expected output:

```
=== CUDA Device Info ===
Device: NVIDIA GeForce RTX 4090
Compute Capability: 8.9
Total Memory: 24.0 GB
SMs: 128
Max Threads/Block: 1024
Shared Memory/Block: 48.0 KB

Hello from CUDA kernel!
CUDA setup verified successfully!
```

---

## What's Next

You now have:

1. ✅ Complete project directory structure
2. ✅ CMake build configuration
3. ✅ Common CUDA utilities and headers
4. ✅ Python package setup
5. ✅ Environment and build scripts

In the next document, we'll dive into **CUDA Foundations** and start building our first kernel.

Continue to: [02_cuda_foundations.md](./02_cuda_foundations.md)

---

## Troubleshooting

### CMake Can't Find CUDA

```bash
# Set CUDA path explicitly
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
cmake -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc ..
```

### pybind11 Not Found

```bash
# Install pybind11 with pip
pip install pybind11

# Or use system package
sudo apt install python3-pybind11
```

### Permission Denied on Scripts

```bash
chmod +x scripts/*.sh
```

### CUDA Version Mismatch

Ensure your PyTorch CUDA version matches your system CUDA:

```bash
# Check system CUDA
nvcc --version

# Check PyTorch CUDA
python -c "import torch; print(torch.version.cuda)"

# If mismatch, reinstall PyTorch with correct version
pip install torch --index-url https://download.pytorch.org/whl/cu121
```
