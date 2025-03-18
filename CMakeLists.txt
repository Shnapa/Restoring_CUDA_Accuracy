cmake_minimum_required(VERSION 3.15)
set(PROJECT_NAME Restoring_Accuracy)
project(${PROJECT_NAME} C CXX)
project(${PROJECT_NAME} CUDA CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_ARCHITECTURES 89)

add_subdirectory(matrix-multiplication)
add_subdirectory(integral)
add_subdirectory(mm_optimized)
add_subdirectory(matrix-multiplication-cuda)
add_subdirectory(cuBLAS_test)