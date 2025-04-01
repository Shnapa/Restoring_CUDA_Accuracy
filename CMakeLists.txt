cmake_minimum_required(VERSION 3.15)
set(CMAKE_CUDA_ARCHITECTURES 75 89)
set(CMAKE_CXX_STANDARD 20)

set(PROJECT_NAME Restoring_Accuracy)
project(${PROJECT_NAME} C CXX)
project(${PROJECT_NAME} CXX CUDA)

set(SOURCES
        include/matrixParser.h
        src/matrixParser.cpp
        include/timeMeasurement.h)

find_package(Threads REQUIRED)
find_package(benchmark REQUIRED)
set(SOURCE_DIRECTORIES include src data)

add_executable(cublasMul        src/cublasMul.cu                    ${SOURCES})
add_executable(cudaMul          src/cudaMul.cu                      ${SOURCES})
add_executable(cudaMulOpt       src/cudaMulOpt.cu                   ${SOURCES})
add_executable(simdMul          src/matrix_simd_mult.cpp            ${SOURCES})
add_executable(simdMulOpt       src/matrix_simd_parallel_mult.cpp   ${SOURCES})
add_executable(matrixMul        matrix-multiplication/main.cpp      ${SOURCES})
add_executable(generateMatrices src/generateMatrices.cpp            ${SOURCES})
add_executable(wmma src/wmma.cu ${SOURCES})

target_include_directories(matrixMul        PRIVATE ${CMAKE_SOURCE_DIR} ${SOURCE_DIRECTORIES})
target_include_directories(simdMul          PRIVATE ${CMAKE_SOURCE_DIR} ${SOURCE_DIRECTORIES})
target_include_directories(simdMulOpt       PRIVATE ${CMAKE_SOURCE_DIR} ${SOURCE_DIRECTORIES})
target_include_directories(cudaMul          PRIVATE ${CMAKE_SOURCE_DIR} ${SOURCE_DIRECTORIES})
target_include_directories(cudaMulOpt       PRIVATE ${CMAKE_SOURCE_DIR} ${SOURCE_DIRECTORIES})
target_include_directories(cublasMul        PRIVATE ${CMAKE_SOURCE_DIR} ${SOURCE_DIRECTORIES})
target_include_directories(generateMatrices PRIVATE ${CMAKE_SOURCE_DIR} ${SOURCE_DIRECTORIES})
target_include_directories(wmma PRIVATE ${CMAKE_SOURCE_DIR} ${SOURCE_DIRECTORIES})

target_link_libraries(cublasMul cublas)
target_link_libraries(generateMatrices  Threads::Threads)
target_link_libraries(cudaMul           benchmark::benchmark)
target_link_libraries(cudaMulOpt        benchmark::benchmark)
target_link_libraries(cublasMul         benchmark::benchmark)
target_link_libraries(simdMul           benchmark::benchmark)
target_link_libraries(simdMulOpt        benchmark::benchmark)

include_directories(src data)

target_compile_options(simdMul      PRIVATE -mavx -mfma -march=native)
target_compile_options(simdMulOpt   PRIVATE -mavx -mfma -march=native)

enable_testing()
add_test(
        NAME Benchmark_CudaMul
        COMMAND $<TARGET_FILE:cudaMul>
        --benchmark_repetitions=10
        --benchmark_color=auto
        --benchmark_time_unit=ms
        --benchmark_out_format=json
        --benchmark_out=../benchmark_results/results_CudaMul.json
)

add_test(
        NAME Benchmark_CudaMulOpt
        COMMAND $<TARGET_FILE:cudaMulOpt>
        --benchmark_repetitions=10
        --benchmark_color=auto
        --benchmark_time_unit=ms
        --benchmark_out_format=json
        --benchmark_out=../benchmark_results/results_CudaMulOpt.json
)

add_test(
        NAME Benchmark_CublasMul
        COMMAND $<TARGET_FILE:cublasMul>
        --benchmark_repetitions=10
        --benchmark_color=auto
        --benchmark_time_unit=ms
        --benchmark_out_format=json
        --benchmark_out=../benchmark_results/results_CublasMul.json
)

add_test(
        NAME Benchmark_SimdMulOpt
        COMMAND $<TARGET_FILE:simdMulOpt>
        --benchmark_repetitions=10
        --benchmark_color=auto
        --benchmark_time_unit=ms
        --benchmark_out_format=json
        --benchmark_out=../benchmark_results/results_SimdMulOpt.json
        --rerun-failed
        --output-on-failure
)

#add_test(
#        NAME Benchmark_SimdMul
#        COMMAND $<TARGET_FILE:simdMul>
#        --benchmark_repetitions=10
#        --benchmark_color=auto
#        --benchmark_time_unit=ms
#        --benchmark_out_format=json
#        --benchmark_out=../benchmark_results/results_SimdMul.json
#)
