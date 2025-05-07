set(SOURCES
        accuracy_comparison.cpp
        accuracy_comparison.h
        ../include/mmul.h
        ../src/cudaMM.cu
        ../src/cublasMM.cu
        ../src/simdMM.cpp
        ../src/wmmaMM.cu
        ../restoring_accuracy/main.cu
)

add_executable(matmul_compare
        main.cu
        ${SOURCES}
)

target_include_directories(matmul_compare PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

target_compile_options(matmul_compare PRIVATE -mavx -mfma -march=native)

find_package(Threads REQUIRED)
target_link_libraries(matmul_compare Threads::Threads cublas ${CUDA_LIBRARIES})
