# CMake generated Testfile for 
# Source directory: /workspace/Docs/Restoring_CUDA_Accuracy
# Build directory: /workspace/Docs/Restoring_CUDA_Accuracy/build
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test([=[Benchmark_CublasMul]=] "/workspace/Docs/Restoring_CUDA_Accuracy/build/cublasTest" "--benchmark_out=../benchmark_results/results_CublasMul.json")
set_tests_properties([=[Benchmark_CublasMul]=] PROPERTIES  _BACKTRACE_TRIPLES "/workspace/Docs/Restoring_CUDA_Accuracy/CMakeLists.txt;128;add_test;/workspace/Docs/Restoring_CUDA_Accuracy/CMakeLists.txt;0;")
add_test([=[Benchmark_CublasHalfMul]=] "/workspace/Docs/Restoring_CUDA_Accuracy/build/cublasHalfTest" "--benchmark_out=../benchmark_results/results_CublasHalfMul.json")
set_tests_properties([=[Benchmark_CublasHalfMul]=] PROPERTIES  _BACKTRACE_TRIPLES "/workspace/Docs/Restoring_CUDA_Accuracy/CMakeLists.txt;132;add_test;/workspace/Docs/Restoring_CUDA_Accuracy/CMakeLists.txt;0;")
add_test([=[Benchmark_WMMA]=] "/workspace/Docs/Restoring_CUDA_Accuracy/build/wmmaTest" "--benchmark_out=../benchmark_results/results_wmma.json")
set_tests_properties([=[Benchmark_WMMA]=] PROPERTIES  _BACKTRACE_TRIPLES "/workspace/Docs/Restoring_CUDA_Accuracy/CMakeLists.txt;136;add_test;/workspace/Docs/Restoring_CUDA_Accuracy/CMakeLists.txt;0;")
add_test([=[Benchmark_CudaMul]=] "/workspace/Docs/Restoring_CUDA_Accuracy/build/cudaMulTest" "--benchmark_out=../benchmark_results/results_CudaMul.json")
set_tests_properties([=[Benchmark_CudaMul]=] PROPERTIES  _BACKTRACE_TRIPLES "/workspace/Docs/Restoring_CUDA_Accuracy/CMakeLists.txt;140;add_test;/workspace/Docs/Restoring_CUDA_Accuracy/CMakeLists.txt;0;")
add_test([=[Benchmark_CudaMulOpt]=] "/workspace/Docs/Restoring_CUDA_Accuracy/build/cudaMulOptTest" "--benchmark_out=../benchmark_results/results_CudaMulOpt_small.json")
set_tests_properties([=[Benchmark_CudaMulOpt]=] PROPERTIES  _BACKTRACE_TRIPLES "/workspace/Docs/Restoring_CUDA_Accuracy/CMakeLists.txt;144;add_test;/workspace/Docs/Restoring_CUDA_Accuracy/CMakeLists.txt;0;")
add_test([=[Benchmark_SIMD]=] "/workspace/Docs/Restoring_CUDA_Accuracy/build/simdMulOptTest" "--benchmark_out=../benchmark_results/results_simd.json")
set_tests_properties([=[Benchmark_SIMD]=] PROPERTIES  _BACKTRACE_TRIPLES "/workspace/Docs/Restoring_CUDA_Accuracy/CMakeLists.txt;148;add_test;/workspace/Docs/Restoring_CUDA_Accuracy/CMakeLists.txt;0;")
