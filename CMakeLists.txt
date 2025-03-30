cmake_minimum_required(VERSION 3.15)
set(CMAKE_CUDA_ARCHITECTURES 75 89)
set(CMAKE_CXX_STANDARD 20)

set(PROJECT_NAME Restoring_Accuracy)
project(${PROJECT_NAME} C CXX)
project(${PROJECT_NAME} CXX CUDA)

set(SOURCES
        include/matrixParser.h
        include/timeMeasurement.h)

set(SOURCE_DIRECTORIES include src data)

add_executable(cublasMul src/cublasMul.cu ${SOURCES})

add_executable(cudaMul src/cudaMul.cu ${SOURCES})

add_executable(cudaMulOpt src/cudaMulOpt.cu ${SOURCES})

add_executable(simdMul src/matrix_simd_mult.cpp ${SOURCES})

add_executable(simdMulOpt src/matrix_simd_parallel_mult.cpp ${SOURCES} )

add_executable(matrixMul matrix-multiplication/main.cpp ${SOURCES})

add_executable(wmma matrix-multiplication/wmma.cu ${SOURCES})

target_include_directories(matrixMul PRIVATE ${CMAKE_SOURCE_DIR} ${SOURCE_DIRECTORIES})

target_include_directories(simdMul PRIVATE ${CMAKE_SOURCE_DIR} ${SOURCE_DIRECTORIES})
target_include_directories(simdMulOpt PRIVATE ${CMAKE_SOURCE_DIR} ${SOURCE_DIRECTORIES})

target_include_directories(cudaMul PRIVATE ${CMAKE_SOURCE_DIR} ${SOURCE_DIRECTORIES})
target_include_directories(cudaMulOpt PRIVATE ${CMAKE_SOURCE_DIR} ${SOURCE_DIRECTORIES})
target_include_directories(cublasMul PRIVATE ${CMAKE_SOURCE_DIR} ${SOURCE_DIRECTORIES})
target_include_directories(wmma PRIVATE ${CMAKE_SOURCE_DIR} ${SOURCE_DIRECTORIES})
target_link_libraries(cublasMul cublas)

include_directories(src data)

target_compile_options(simdMul PRIVATE -mavx -mfma -march=native)
target_compile_options(simdMulOpt PRIVATE -mavx -mfma -march=native)
