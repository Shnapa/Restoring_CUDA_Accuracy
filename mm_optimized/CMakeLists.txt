add_executable(MatrixMultiplication_Optimized
        matrix_mul.cu)

target_include_directories(MatrixMultiplication_Optimized PRIVATE ${CMAKE_SOURCE_DIR})

