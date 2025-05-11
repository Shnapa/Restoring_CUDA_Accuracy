set(MATMUL_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/main.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/accuracy_comparison.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/matrixParser.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/cudaMM.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/cublasMM.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/simdMM.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/wmmaMM.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/../restoring_accuracy/restore.cu
)

add_executable(matmul_compare ${MATMUL_SOURCES})

target_include_directories(matmul_compare PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/..
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
    ${CMAKE_CURRENT_SOURCE_DIR}/../src
)

target_compile_options(matmul_compare PRIVATE -mavx -mfma -march=native)

find_package(Threads REQUIRED)
target_link_libraries(matmul_compare Threads::Threads cublas ${CUDA_LIBRARIES})
