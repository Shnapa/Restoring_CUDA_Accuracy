cmake_minimum_required(VERSION 3.15)
set(CMAKE_CUDA_ARCHITECTURES 75 89)
set(CMAKE_CXX_STANDARD 20)

set(PROJECT_NAME Restoring_Accuracy)
project(${PROJECT_NAME} C CXX)
project(${PROJECT_NAME} CXX CUDA)

set(SOURCE_DIRECTORIES include src data)
set(SOURCES
        include/matrixParser.h
        src/matrixParser.cpp
        include/timeMeasurement.h)

find_package(Threads REQUIRED)
find_package(benchmark REQUIRED)

##############################################
###              Default run               ###
##############################################

add_executable(cublasMul        src/cublasMul.cu                       ${SOURCES})
add_executable(cudaMul          src/cudaMul.cu                      ${SOURCES})
add_executable(cudaMulOpt       src/cudaMulOpt.cu                   ${SOURCES})
add_executable(wmma             src/wmma.cu                         ${SOURCES})
add_executable(simdMul          src/matrix_simd_mult.cpp            ${SOURCES})
add_executable(simdMulOpt       src/matrix_simd_parallel_mult.cpp   ${SOURCES})
add_executable(generateMatrices src/generateMatrices.cpp            ${SOURCES})

foreach(target cublasMul cudaMul cudaMulOpt wmma simdMul simdMulOpt generateMatrices)
    target_include_directories(${target} PRIVATE ${CMAKE_SOURCE_DIR} ${SOURCE_DIRECTORIES})
endforeach()

target_link_libraries(cublasMul cublas)
target_link_libraries(generateMatrices  Threads::Threads)

target_compile_options(simdMul      PRIVATE -mavx -mfma -march=native)
target_compile_options(simdMulOpt   PRIVATE -mavx -mfma -march=native)

#add_subdirectory(testing)

##############################################
###                 BM run                 ###
##############################################

# add_executable(cublasTest           benchmarks/cublasMulBM.cu   ${SOURCES})
#add_executable(cudaMulTest          benchmarks/cudaMulBM.cu     ${SOURCES})
#add_executable(cudaMulOptTest       benchmarks/cudaMulOptBM.cu  ${SOURCES})
#add_executable(simdMulOptTest       benchmarks/simdMulOptBM.cpp ${SOURCES})
#add_executable(wmmaTest             benchmarks/wmmaBM.cu        ${SOURCES})

#target_link_libraries(cublasTest benchmark::benchmark cublas)
#target_compile_options(simdMulOptTest   PRIVATE -mavx -mfma -march=native)
#foreach(target cudaMulTest cudaMulOptTest simdMulOptTest wmmaTest)
 #   target_include_directories(${target} PRIVATE ${CMAKE_SOURCE_DIR} ${SOURCE_DIRECTORIES})
 #   target_link_libraries(${target} PRIVATE benchmark::benchmark)
#endforeach()

##############################################
###                 Tests                  ###
##############################################
#enable_testing()
#add_test(
#        NAME Benchmark_CudaMul
#        COMMAND $<TARGET_FILE:cudaMulTest>
#        --benchmark_repetitions=5
#        --benchmark_color=auto
#        --benchmark_time_unit=ms
#        --benchmark_report_aggregates_only
#        --benchmark_out_format=json
#        --benchmark_out=../benchmark_results/results_CudaMul.json
#)

#add_test(
#        NAME Benchmark_CudaMulOpt
#        COMMAND $<TARGET_FILE:cudaMulOptTest>
#        --benchmark_repetitions=2
#        --benchmark_color=auto
#        --benchmark_time_unit=ms
#        --benchmark_report_aggregates_only
#        --benchmark_out_format=json
#        --benchmark_out=../benchmark_results/results_CudaMulOpt_small.json
#)

#add_test(
#        NAME Benchmark_CublasMul
#        COMMAND $<TARGET_FILE:cublasTest>
#        --benchmark_repetitions=5
#        --benchmark_color=auto
#        --benchmark_time_unit=ms
#        --benchmark_report_aggregates_only
#        --benchmark_out_format=json
#        --benchmark_out=../benchmark_results/results_CublasMul.json
#)

#add_test(
#        NAME Benchmark_SimdMulOpt
#        COMMAND $<TARGET_FILE:simdMulOptTest>
#        --benchmark_repetitions=2
#        --benchmark_color=auto
#        --benchmark_time_unit=ms
#        --benchmark_report_aggregates_only
#        --benchmark_out_format=json
#        --benchmark_out=../benchmark_results/results_SimdMulOpt_small.json
#)

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
