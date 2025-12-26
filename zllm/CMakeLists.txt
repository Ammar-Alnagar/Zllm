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
