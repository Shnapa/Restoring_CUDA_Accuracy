add_executable(cublasTest
        cublasTest.cu)

target_include_directories(cublasTest PRIVATE ${CMAKE_SOURCE_DIR})
target_link_libraries(cublasTest cublas)
