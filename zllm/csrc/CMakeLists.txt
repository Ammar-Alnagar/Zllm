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
)

# Memory management sources
set(MEMORY_SOURCES
    memory/kv_cache.cu
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
