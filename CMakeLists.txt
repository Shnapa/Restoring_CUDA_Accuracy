cmake_minimum_required(VERSION 3.15)
set(CMAKE_CUDA_ARCHITECTURES 75 89)
set(CMAKE_CXX_STANDARD 20)

set(PROJECT_NAME Restoring_Accuracy)
project(${PROJECT_NAME} C CXX CUDA)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --fmad=false --prec-div=true --prec-sqrt=true --ftz=false")

set(SOURCE_DIRECTORIES include src data restoring_accuracy utils)

set(SOURCES
        src/matrixParser.cpp
        include/matrixParser.h

        include/timeMeasurement.h

#        src/compare.cu

        include/mmul.h
)

find_package(Threads REQUIRED)
find_package(benchmark REQUIRED)

##############################################
###              Default run               ###
##############################################

#add_subdirectory(utils)

add_executable(restoringAcc     restoring_accuracy/main.cu                                          ${SOURCES} )
add_executable(cublasMul        mains/cublasMain.cu                 src/cublasMM.cu                 ${SOURCES} )
add_executable(cudaMul          mains/cudaMain.cu                   src/cudaMM.cu                   ${SOURCES} )
add_executable(cudaMulOpt       mains/cudaMulOptMain.cu             src/cudaMM.cu                   ${SOURCES} )
add_executable(wmma             mains/wmmaMain.cu                   src/wmmaMM.cu                   ${SOURCES} )
add_executable(simdMulOpt mains/matrix_simd_parallel_mult.cpp src/simdMM.cpp src/cublasMM.cu         ${SOURCES} )
#add_executable(generateMatrices src/generateMatrices.cpp                                            ${SOURCES} )

foreach(target cublasMul cudaMul cudaMulOpt wmma simdMulOpt restoringAcc)
    target_include_directories(${target} PRIVATE ${CMAKE_SOURCE_DIR} ${SOURCE_DIRECTORIES})
endforeach()

foreach(target cublasMul cudaMul cudaMulOpt wmma simdMulOpt restoringAcc)
    target_link_libraries(${target} cublas)
endforeach()


set(SOURCES
        utils/accuracy_comparison.cpp
        utils/accuracy_comparison.h
        include/mmul.h
        src/cudaMM.cu
        src/cublasMM.cu
        src/simdMM.cpp
        src/wmmaMM.cu
        restoring_accuracy/restore.cu
)

add_executable(matmul_compare
        utils/main.cu
        ${SOURCES}
)

target_include_directories(matmul_compare PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

target_compile_options(matmul_compare PRIVATE -mavx -mfma -march=native)

find_package(Threads REQUIRED)
target_link_libraries(matmul_compare Threads::Threads cublas ${CUDA_LIBRARIES})

#target_link_libraries(generateMatrices  Threads::Threads)

target_compile_options(simdMulOpt      PRIVATE -mavx -mfma -march=native)
##############################################
###                 BM run                 ###
##############################################

#add_executable(cublasTest           benchmarks/cublasMulBM.cu   ${SOURCES})
#add_executable(cudaMulTest          benchmarks/cudaMulBM.cu     ${SOURCES})
#add_executable(cudaMulOptTest       benchmarks/cudaMulOptBM.cu  ${SOURCES})
#add_executable(simdMulOptTest       benchmarks/simdMulOptBM.cpp ${SOURCES})
#add_executable(wmmaTest             benchmarks/wmmaBM.cu        ${SOURCES})
#
#target_link_libraries(cublasTest benchmark::benchmark cublas)
#target_compile_options(simdMulOptTest   PRIVATE -mavx -mfma -march=native)
#
#foreach(target cudaMulTest cudaMulOptTest simdMulOptTest wmmaTest)
#    target_include_directories(${target} PRIVATE ${CMAKE_SOURCE_DIR} ${SOURCE_DIRECTORIES})
#    target_link_libraries(${target} PRIVATE benchmark::benchmark)
#endforeach()

##############################################
###                 Tests                  ###
##############################################
#enable_testing()
#add_test(
#        NAME Benchmark_CudaMul
#        COMMAND $<TARGET_FILE:cudaMulTest>
#        --benchmark_repetitions=2
#        --benchmark_color=auto
#        --benchmark_time_unit=ms
#        --benchmark_report_aggregates_only
#        --benchmark_out_format=json
#        --benchmark_out=../benchmark_results/results_CudaMul.json
#)
#
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
#
#add_test(
#        NAME Benchmark_CublasMul
#        COMMAND $<TARGET_FILE:cublasTest>
#        --benchmark_repetitions=2
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
