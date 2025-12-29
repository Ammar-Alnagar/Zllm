// =============================================================================
// memory_utils.cu - Memory Utility Functions
// =============================================================================

#include "memory_pool.cuh"
#include <cuda_runtime.h>

namespace mini_vllm {

/**
 * Get GPU memory info
 */
void get_gpu_memory_info(size_t& free_bytes, size_t& total_bytes) {
    CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));
}

/**
 * Print detailed GPU memory information
 */
void print_gpu_memory_info() {
    size_t free_bytes, total_bytes;
    get_gpu_memory_info(free_bytes, total_bytes);

    float free_gb = free_bytes / (1024.0f * 1024.0f * 1024.0f);
    float total_gb = total_bytes / (1024.0f * 1024.0f * 1024.0f);
    float used_gb = total_gb - free_gb;

    printf("=== GPU Memory Info ===\n");
    printf("Total: %.2f GB\n", total_gb);
    printf("Used:  %.2f GB (%.1f%%)\n", used_gb, (used_gb / total_gb) * 100);
    printf("Free:  %.2f GB\n", free_gb);
}

/**
 * Calculate maximum KV cache size based on available memory
 *
 * @param model_size_gb: Model weight size in GB
 * @param activation_gb: Reserved for activations
 * @param num_kv_heads: Number of KV heads
 * @param head_dim: Head dimension
 * @param block_size: Tokens per block
 * @param dtype_size: Bytes per element
 * @return: Number of blocks that can be allocated
 */
int calculate_max_kv_blocks(
    float model_size_gb,
    float activation_gb,
    int num_kv_heads,
    int head_dim,
    int block_size,
    int dtype_size
) {
    size_t free_bytes, total_bytes;
    get_gpu_memory_info(free_bytes, total_bytes);

    // Reserve some headroom
    float headroom_gb = 0.5f;

    // Available for KV cache
    float available_gb = (free_bytes / 1e9f) - model_size_gb - activation_gb - headroom_gb;

    if (available_gb < 0) {
        printf("Warning: Not enough GPU memory!\n");
        return 0;
    }

    size_t available_bytes = static_cast<size_t>(available_gb * 1e9);

    // Block size in bytes
    size_t block_bytes = 2 * block_size * num_kv_heads * head_dim * dtype_size;

    int max_blocks = available_bytes / block_bytes;

    printf("Available for KV cache: %.2f GB\n", available_gb);
    printf("Block size: %zu bytes\n", block_bytes);
    printf("Max blocks: %d\n", max_blocks);
    printf("Max tokens: %d\n", max_blocks * block_size);

    return max_blocks;
}

/**
 * Memory-efficient buffer for temporary allocations
 */
class ScratchBuffer {
public:
    ScratchBuffer() : data_(nullptr), size_(0) {}

    ~ScratchBuffer() {
        free();
    }

    void* get(size_t required_size) {
        if (required_size > size_) {
            free();
            CUDA_CHECK(cudaMalloc(&data_, required_size));
            size_ = required_size;
        }
        return data_;
    }

    void free() {
        if (data_) {
            cudaFree(data_);
            data_ = nullptr;
            size_ = 0;
        }
    }

    size_t size() const { return size_; }

private:
    void* data_;
    size_t size_;
};

// Global scratch buffer (one per stream ideally)
static thread_local ScratchBuffer g_scratch_buffer;

void* get_scratch_buffer(size_t size) {
    return g_scratch_buffer.get(size);
}

} // namespace mini_vllm