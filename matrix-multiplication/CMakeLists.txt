add_executable(MM
        main.cpp
        ../src/matrixParser.cpp
        ../include/matrixParser.h
)

target_include_directories(MM PRIVATE ${CMAKE_SOURCE_DIR})

