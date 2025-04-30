set(SOURCES
        accuracy_comparison.cpp
        accuracy_comparison.h
)

add_executable(matmul_compare
        main.cpp
        ${SOURCES}
)

target_include_directories(matmul_compare PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

target_compile_options(matmul_compare PRIVATE -mavx -mfma -march=native)

find_package(Threads REQUIRED)
target_link_libraries(matmul_compare Threads::Threads)
