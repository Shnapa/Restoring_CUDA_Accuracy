cmake_minimum_required(VERSION 3.29)
project(matrix_testing)

set(CMAKE_CXX_STANDARD 14)

include_directories(.)

add_executable(matrix_testing
    test_main.cpp
    testing.cpp
    testing.h)
