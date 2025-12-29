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