add_executable(MatrixMultiplicationCuda
        matrix_multiplication_cuda.cu)

target_include_directories(MatrixMultiplicationCuda PRIVATE ${CMAKE_SOURCE_DIR})

