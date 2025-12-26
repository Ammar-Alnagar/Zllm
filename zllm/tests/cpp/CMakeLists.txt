cmake_minimum_required(VERSION 3.24 FATAL_ERROR)

project(cuda_test LANGUAGES CXX CUDA)

find_package(CUDAToolkit REQUIRED)

add_executable(test_cuda test_kernels.cu)

target_link_libraries(test_cuda CUDA::cudart)

set_target_properties(test_cuda PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)