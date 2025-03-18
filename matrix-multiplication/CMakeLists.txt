add_executable(MatrixMultiplication
        main.cpp
        matrix_multiplication.cpp
        matrix_multiplication.h)

target_include_directories(MatrixMultiplication PRIVATE ${CMAKE_SOURCE_DIR})

