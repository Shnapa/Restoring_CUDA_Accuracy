add_executable(MM
        main.cpp
        matrix_multiplication.cpp
        matrix_multiplication.h)

target_include_directories(MM PRIVATE ${CMAKE_SOURCE_DIR})

