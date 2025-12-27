cmake_minimum_required(VERSION 3.24 FATAL_ERROR)

project(cuda_test LANGUAGES CXX CUDA)

find_package(CUDAToolkit REQUIRED)

add_executable(test_kernels test_kernels.cu)
add_executable(test_rmsnorm test_rmsnorm.cu)

target_link_libraries(test_kernels CUDA::cudart stdc++ m)
target_link_libraries(test_rmsnorm CUDA::cudart stdc++ m)

set_target_properties(test_kernels PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)
set_target_properties(test_rmsnorm PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

# Set CUDA architecture flags for compatibility
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode arch=compute_89,code=sm_89")