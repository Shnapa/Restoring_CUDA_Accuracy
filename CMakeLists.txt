cmake_minimum_required(VERSION 3.15)
set(CMAKE_CUDA_ARCHITECTURES 75 89)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(PROJECT_NAME Restoring_Accuracy)
project(${PROJECT_NAME} C CXX)
project(${PROJECT_NAME} CXX CUDA)

set(SOURCES
        include/matrixParser.h
        src/matrixParser.cpp
        include/timeMeasurement.h
        matrix-multiplication/multBrut.h)
include_directories(src data)

find_package(Threads REQUIRED)
find_package(benchmark REQUIRED)
set(SOURCE_DIRECTORIES include src data)

##############################################
###              Default run               ###
##############################################
add_executable(cublasMul        src/cublasMul.cu                    ${SOURCES})
add_executable(cudaMul          src/cudaMul.cu                      ${SOURCES})
add_executable(cudaMulOpt       src/cudaMulOpt.cu                   ${SOURCES})
add_executable(simdMul          src/matrix_simd_mult.cpp            ${SOURCES})
add_executable(simdMulOpt       src/matrix_simd_parallel_mult.cpp   ${SOURCES})
add_executable(generateMatrices src/generateMatrices.cpp            ${SOURCES})
add_executable(wmma             src/wmma.cu                         ${SOURCES})

target_include_directories(simdMul          PRIVATE ${CMAKE_SOURCE_DIR} ${SOURCE_DIRECTORIES})
target_include_directories(simdMulOpt       PRIVATE ${CMAKE_SOURCE_DIR} ${SOURCE_DIRECTORIES})
target_include_directories(cudaMul          PRIVATE ${CMAKE_SOURCE_DIR} ${SOURCE_DIRECTORIES})
target_include_directories(cudaMulOpt       PRIVATE ${CMAKE_SOURCE_DIR} ${SOURCE_DIRECTORIES})
target_include_directories(cublasMul        PRIVATE ${CMAKE_SOURCE_DIR} ${SOURCE_DIRECTORIES})
target_include_directories(generateMatrices PRIVATE ${CMAKE_SOURCE_DIR} ${SOURCE_DIRECTORIES})
target_include_directories(wmma             PRIVATE ${CMAKE_SOURCE_DIR} ${SOURCE_DIRECTORIES})

target_link_libraries(cublasMul cublas)
target_link_libraries(generateMatrices  Threads::Threads)

target_compile_options(simdMul      PRIVATE -mavx -mfma -march=native)
target_compile_options(simdMulOpt   PRIVATE -mavx -mfma -march=native)

#add_subdirectory(testing)

##############################################
###                 BM run                 ###
##############################################

add_executable(cublasTest           benchmarks/cublasMulBM.cu   ${SOURCES})
add_executable(cudaMulTest          benchmarks/cudaMulBM.cu     ${SOURCES})
add_executable(cudaMulOptTest       benchmarks/cudaMulOptBM.cu  ${SOURCES})
add_executable(simdMulOptTest       benchmarks/simdMulOptBM.cpp ${SOURCES})
add_executable(wmmaTest             benchmarks/wmmaBM.cu        ${SOURCES})

target_link_libraries(cublasTest            benchmark::benchmark cublas)
target_link_libraries(cudaMulTest           benchmark::benchmark)
target_link_libraries(cudaMulOptTest        benchmark::benchmark)
target_link_libraries(simdMulOptTest        benchmark::benchmark)
target_link_libraries(wmmaTest              benchmark::benchmark)

target_compile_options(simdMulOptTest   PRIVATE -mavx -mfma -march=native)

target_include_directories(cublasTest           PRIVATE ${CMAKE_SOURCE_DIR} ${SOURCE_DIRECTORIES})
target_include_directories(cudaMulTest          PRIVATE ${CMAKE_SOURCE_DIR} ${SOURCE_DIRECTORIES})
target_include_directories(cudaMulOptTest       PRIVATE ${CMAKE_SOURCE_DIR} ${SOURCE_DIRECTORIES})
target_include_directories(simdMulOptTest       PRIVATE ${CMAKE_SOURCE_DIR} ${SOURCE_DIRECTORIES})
target_include_directories(wmmaTest             PRIVATE ${CMAKE_SOURCE_DIR} ${SOURCE_DIRECTORIES})

##############################################
###                 Tests                  ###
##############################################
enable_testing()
add_test(
        NAME Benchmark_CudaMul
        COMMAND $<TARGET_FILE:cudaMulTest>
        --benchmark_repetitions=5
        --benchmark_color=auto
        --benchmark_time_unit=ms
        --benchmark_report_aggregates_only
        --benchmark_out_format=json
        --benchmark_out=../benchmark_results/results_CudaMul.json
)

add_test(
        NAME Benchmark_CudaMulOpt
        COMMAND $<TARGET_FILE:cudaMulOptTest>
        --benchmark_repetitions=2
        --benchmark_color=auto
        --benchmark_time_unit=ms
        --benchmark_report_aggregates_only
        --benchmark_out_format=json
        --benchmark_out=../benchmark_results/results_CudaMulOpt_small.json
)

add_test(
        NAME Benchmark_CublasMul
        COMMAND $<TARGET_FILE:cublasTest>
        --benchmark_repetitions=5
        --benchmark_color=auto
        --benchmark_time_unit=ms
        --benchmark_report_aggregates_only
        --benchmark_out_format=json
        --benchmark_out=../benchmark_results/results_CublasMul.json
)

add_test(
        NAME Benchmark_SimdMulOpt
        COMMAND $<TARGET_FILE:simdMulOptTest>
        --benchmark_repetitions=2
        --benchmark_color=auto
        --benchmark_time_unit=ms
        --benchmark_report_aggregates_only
        --benchmark_out_format=json
        --benchmark_out=../benchmark_results/results_SimdMulOpt_small.json
)

#add_test(
#        NAME Benchmark_wmmaMul
#        COMMAND $<TARGET_FILE:wmmaTest>
#        --benchmark_repetitions=5
#        --benchmark_color=auto
#        --benchmark_time_unit=ms
#        --benchmark_report_aggregates_only
#        --benchmark_out_format=json
#        --benchmark_out=../benchmark_results/results_wmmaMul.json
#)