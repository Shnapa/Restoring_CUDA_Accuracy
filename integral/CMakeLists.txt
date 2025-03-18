add_executable(integral
        integral.cu
)

target_include_directories(integral PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})