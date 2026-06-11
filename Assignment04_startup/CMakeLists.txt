cmake_minimum_required(VERSION 3.24)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CUDA_ARCHITECTURES native)
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)

project(CudaLab VERSION 0.1 LANGUAGES CXX CUDA)

find_package(CUDAToolkit)
include_directories(${CUDAToolkit_INCLUDE_DIRS})

add_executable(Assignment04 Assignment04.cu)
