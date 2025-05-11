cmake_minimum_required(VERSION 3.24)
set(CMAKE_CUDA_ARCHITECTURES native)
set(CMAKE_CXX_STANDARD 20)

set(PROJECT_NAME Restoring_Accuracy)
project(${PROJECT_NAME} C CXX CUDA)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --fmad=false --prec-div=true --prec-sqrt=true --ftz=false")

set(SOURCE_DIRECTORIES include src data restoring_accuracy utils)

set(SOURCES
        src/matrixParser.cu
        include/matrixParser.h
        include/timeMeasurement.h

        include/compareMM.h
        src/compare.cpp
)

find_package(Threads REQUIRED)
find_package(benchmark REQUIRED)

##############################################
###              Default run               ###
##############################################


add_executable(restoringAcc
        restoring_accuracy/restoreMain.cu

        include/mmul.cuh
        ${SOURCES})

add_executable(cublasMul
        mains/cublasMain.cu
        matrix-multiplication/cublasMM.cu

        include/mmul.cuh
        ${SOURCES} )

<<<<<<< Updated upstream
add_executable(cudaMul
        mains/cudaMain.cu
        matrix-multiplication/cudaMM.cu

        include/mmul.cuh
        ${SOURCES})

add_executable(cudaMulOpt
        mains/cudaMulOptMain.cu
        matrix-multiplication/cudaMM.cu

        include/mmul.cuh
        ${SOURCES} )

add_executable(wmma
        mains/wmmaMain.cu
        matrix-multiplication/wmmaMM.cu

        include/mmul.cuh
        ${SOURCES} )

add_executable(simdMulOpt
        mains/simdMulOptMain.cpp

        include/mmul.cuh
        ${SOURCES} )

add_executable(generateMatrices
        src/generateMatrices.cpp
        ${SOURCES} )

add_executable(matmul_compare
        utils/main.cu
        utils/accuracy_comparison.cpp
        utils/accuracy_comparison.h

        include/mmul.cuh

        matrix-multiplication/cudaMM.cu
        matrix-multiplication/cublasMM.cu

        include/simdMM.h

        matrix-multiplication/wmmaMM.cu
        restoring_accuracy/restore.cu
        ${SOURCES}
)


foreach(target cublasMul cudaMul cudaMulOpt wmma simdMulOpt restoringAcc matmul_compare generateMatrices)
    target_include_directories(${target} PRIVATE ${CMAKE_SOURCE_DIR} ${SOURCE_DIRECTORIES})
endforeach()

foreach(target cublasMul wmma restoringAcc)
    target_link_libraries(${target} cublas)
endforeach()
=======
#target_link_libraries(generateMatrices  Threads::Threads)
>>>>>>> Stashed changes

target_compile_options(simdMulOpt      PRIVATE -mavx -mfma -march=native)
target_compile_options(matmul_compare PRIVATE -mavx -mfma -march=native)
target_link_libraries(matmul_compare Threads::Threads cublas ${CUDA_LIBRARIES})
target_link_libraries(generateMatrices  Threads::Threads)
##############################################
###                 BM run                 ###
##############################################

add_executable(cublasTest
        benchmarks/cublasMulBM.cu
        ${SOURCES})

add_executable(cublasHalfTest
        benchmarks/cublasHalfBM.cu
        ${SOURCES})

add_executable(cudaMulTest
        benchmarks/cudaMulBM.cu
        ${SOURCES})

add_executable(cudaMulOptTest
        benchmarks/cudaMulOptBM.cu
        ${SOURCES})

add_executable(simdMulOptTest
        benchmarks/simdMulOptBM.cpp
        ${SOURCES})

add_executable(wmmaTest
        benchmarks/wmmaBM.cu
        ${SOURCES})

target_compile_options(simdMulOptTest   PRIVATE -mavx -mfma -march=native)

foreach(target cublasTest cublasHalfTest cudaMulTest cudaMulOptTest simdMulOptTest wmmaTest)
    target_include_directories(${target} PRIVATE ${CMAKE_SOURCE_DIR} ${SOURCE_DIRECTORIES})
    if("${target}" STREQUAL "cublasTest" OR "${target}" STREQUAL "cublasHalfTest")
        target_link_libraries(${target} PRIVATE benchmark::benchmark cublas)
    else ()
        target_link_libraries(${target} PRIVATE benchmark::benchmark)
    endif ()
endforeach()

##############################################
###                 Tests                  ###
##############################################
enable_testing()
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
        NAME Benchmark_CublasHalfMul
        COMMAND $<TARGET_FILE:cublasHalfTest>
        --benchmark_repetitions=5
        --benchmark_color=auto
        --benchmark_time_unit=ms
        --benchmark_report_aggregates_only
        --benchmark_out_format=json
        --benchmark_out=../benchmark_results/results_CublasHalfMul.json
)

<<<<<<< Updated upstream
add_test(
        NAME Benchmark_WMMA
        COMMAND $<TARGET_FILE:wmmaTest>
        --benchmark_repetitions=5
        --benchmark_color=auto
        --benchmark_time_unit=ms
        --benchmark_report_aggregates_only
        --benchmark_out_format=json
        --benchmark_out=../benchmark_results/results_wmma.json
)

add_test(
        NAME Benchmark_CudaMulOpt
        COMMAND $<TARGET_FILE:cudaMulOptTest>
        --benchmark_repetitions=5
        --benchmark_color=auto
        --benchmark_time_unit=ms
        --benchmark_report_aggregates_only
        --benchmark_out_format=json
        --benchmark_out=../benchmark_results/results_CudaMulOpt_small.json
)

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
        NAME Benchmark_SIMD
        COMMAND $<TARGET_FILE:simdMulOptTest>
        --benchmark_repetitions=5
        --benchmark_color=auto
        --benchmark_time_unit=ms
        --benchmark_report_aggregates_only
        --benchmark_out_format=json
        --benchmark_out=../benchmark_results/results_simd.json
)
=======
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

add_executable(generate_matrices src/generateMatrices.cpp)

target_include_directories(generate_matrices PRIVATE ${CMAKE_SOURCE_DIR} ${SOURCE_DIRECTORIES})
target_compile_options(generate_matrices PRIVATE -O2)

add_subdirectory(utils)
>>>>>>> Stashed changes
