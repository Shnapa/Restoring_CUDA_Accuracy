cmake_minimum_required(VERSION 3.24)
project(Restoring_Accuracy C CXX CUDA)

set(CMAKE_CUDA_ARCHITECTURES native)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --fmad=false --prec-div=true --prec-sqrt=true --ftz=false")

find_package(Threads REQUIRED)
find_package(benchmark REQUIRED)

set(SOURCE_DIRECTORIES include src data restoring_accuracy utils matrix-multiplication mains benchmarks)
include_directories(${SOURCE_DIRECTORIES})

set(SOURCES
    src/matrixParser.cu
    src/compare.cpp
    include/matrixParser.h
    include/timeMeasurement.h
    include/compareMM.h
)

##############################################
###              Default run               ###
##############################################

add_executable(restoringAcc
    restoring_accuracy/restoreMain.cu
    include/mmul.cuh
    ${SOURCES}
)

add_executable(cublasMul
    mains/cublasMain.cu
    matrix-multiplication/cublasMM.cu
    include/mmul.cuh
    ${SOURCES}
)

add_executable(cudaMul
    mains/cudaMain.cu
    matrix-multiplication/cudaMM.cu
    include/mmul.cuh
    ${SOURCES}
)

add_executable(cudaMulOpt
    mains/cudaMulOptMain.cu
    matrix-multiplication/cudaMM.cu
    include/mmul.cuh
    ${SOURCES}
)

add_executable(wmma
    mains/wmmaMain.cu
    matrix-multiplication/wmmaMM.cu
    include/mmul.cuh
    ${SOURCES}
)

add_executable(simdMulOpt
    mains/simdMulOptMain.cpp
    include/mmul.cuh
    ${SOURCES}
)

add_executable(generateMatrices
    src/generateMatrices.cpp
    ${SOURCES}
)

add_executable(matmul_compare
    utils/main.cu
    utils/accuracy_comparison.cpp
    matrix-multiplication/cudaMM.cu
    matrix-multiplication/cublasMM.cu
    matrix-multiplication/wmmaMM.cu
    restoring_accuracy/restore.cu
    include/mmul.cuh
    include/simdMM.h
    ${SOURCES}
)

# Include and link
foreach(target cublasMul cudaMul cudaMulOpt wmma simdMulOpt restoringAcc matmul_compare generateMatrices)
    target_include_directories(${target} PRIVATE ${SOURCE_DIRECTORIES})
endforeach()

foreach(target cublasMul wmma restoringAcc matmul_compare)
    target_link_libraries(${target} PRIVATE cublas)
endforeach()

target_link_libraries(matmul_compare PRIVATE Threads::Threads ${CUDA_LIBRARIES})
target_link_libraries(generateMatrices PRIVATE Threads::Threads)

target_compile_options(simdMulOpt PRIVATE -mavx -mfma -march=native)
target_compile_options(matmul_compare PRIVATE -mavx -mfma -march=native)

##############################################
###                 BM run                 ###
##############################################

add_executable(cublasTest benchmarks/cublasMulBM.cu ${SOURCES})
add_executable(cublasHalfTest benchmarks/cublasHalfBM.cu ${SOURCES})
add_executable(cudaMulTest benchmarks/cudaMulBM.cu ${SOURCES})
add_executable(cudaMulOptTest benchmarks/cudaMulOptBM.cu ${SOURCES})
add_executable(simdMulOptTest benchmarks/simdMulOptBM.cpp ${SOURCES})
add_executable(wmmaTest benchmarks/wmmaBM.cu ${SOURCES})
add_executable(wmmaRestoreTest benchmarks/wmmaRestoreBM.cu ${SOURCES})

target_compile_options(simdMulOptTest PRIVATE -mavx -mfma -march=native)

foreach(target cublasTest cublasHalfTest cudaMulTest cudaMulOptTest simdMulOptTest wmmaTest wmmaRestoreTest)
    target_include_directories(${target} PRIVATE ${SOURCE_DIRECTORIES})
    if("${target}" MATCHES "cublas.*")
        target_link_libraries(${target} PRIVATE benchmark::benchmark cublas)
    else()
        target_link_libraries(${target} PRIVATE benchmark::benchmark)
    endif()
endforeach()

##############################################
###                 Tests                  ###
##############################################

enable_testing()
#add_test(
#        NAME Benchmark_CublasMul
#        COMMAND $<TARGET_FILE:cublasTest>
#        --benchmark_repetitions=2
#        --benchmark_color=auto
#        --benchmark_time_unit=ms
#        --benchmark_report_aggregates_only
#        --benchmark_out_format=json
#        --benchmark_out=../benchmark_results/results_CublasMul_0.json
#)
#
add_test(
        NAME Benchmark_CublasHalfMul
        COMMAND $<TARGET_FILE:cublasHalfTest>
        --benchmark_repetitions=2
        --benchmark_color=auto
        --benchmark_time_unit=ms
        --benchmark_report_aggregates_only
        --benchmark_out_format=json
        --benchmark_out=../benchmark_results/results_CublasHalfMul_3.json
)
#
#add_test(
#        NAME Benchmark_WMMA
#        COMMAND $<TARGET_FILE:wmmaTest>
#        --benchmark_repetitions=5
#        --benchmark_color=auto
#        --benchmark_time_unit=ms
#        --benchmark_report_aggregates_only
#        --benchmark_out_format=json
#        --benchmark_out=../benchmark_results/results_wmma.json
#)
#
add_test(
        NAME Benchmark_CudaMulOpt
        COMMAND $<TARGET_FILE:cudaMulOptTest>
        --benchmark_repetitions=2
        --benchmark_color=auto
        --benchmark_time_unit=ms
        --benchmark_report_aggregates_only
        --benchmark_out_format=json
        --benchmark_out=../benchmark_results/results_CudaMulOpt_3.json
)
#
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
#
#add_test(
#        NAME Benchmark_SIMD
#        COMMAND $<TARGET_FILE:simdMulOptTest>
#        --benchmark_repetitions=5
#        --benchmark_color=auto
#        --benchmark_time_unit=ms
#        --benchmark_report_aggregates_only
#        --benchmark_out_format=json
#        --benchmark_out=../benchmark_results/results_simd.json
#)

add_test(
        NAME Benchmark_WMMARestore
        COMMAND $<TARGET_FILE:wmmaRestoreTest>
        --benchmark_repetitions=2
        --benchmark_color=auto
        --benchmark_time_unit=ms
        --benchmark_report_aggregates_only
        --benchmark_out_format=json
        --benchmark_out=../benchmark_results/results_wmmaRestore_3.json
)