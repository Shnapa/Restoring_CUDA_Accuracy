add_executable(MMCUDA
        matrix_multiplication_cuda.cu)

target_include_directories(MMCUDA PRIVATE ${CMAKE_SOURCE_DIR})

