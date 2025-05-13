# Restoring the Accuracy of Calculations after Operations on CUDA Tensor Cores

### Aim of the project:
This project makes NVIDIA Tensor Core math more accurate without slowing it down. We use simple error-compensation and mixed-precision tricks to fix rounding errors, then compare our results and speed against regular matrix multiply. The goal is fast, reliable GPU computing of matrices, without loosing accuracy of computations.  

### CUDA & SIMD matrix multiplications

We explored several methods for matrix multiplication using CUDA and SIMD:

- **Default CUDA Multiplication**: A basic implementation using CUDA C++ kernels. This serves as a reference point for performance and accuracy, demonstrating how manual control over threads and memory can be used for matrix operations.

- **cuBLAS Multiplication**: Utilizes the `cublasGemmEx` function from the cuBLAS library to perform mixed-precision matrix multiplication. This approach automatically leverages Tensor Cores when possible, offering a highly optimized and production-grade solution.

- **WMMA (Warp Matrix Multiply-Accumulate)**: A low-level API that gives fine-grained control over Tensor Core operations. Using WMMA, we explicitly map thread warps to matrix tiles and invoke Tensor Core instructions directly, enabling experimentation with precision restoration techniques at the lowest level.

- **SIMD Multiplication**: To provide a performance and accuracy baseline, we implemented a SIMD (Single Instruction, Multiple Data) multiplication on CPU. While it is not CUDA-based, it helps highlight the trade-offs between CPU and GPU performance and numerical precision.

### Brief explanation of each algorithm's main features:
  - ##### cuBLAS Half-Precision (`cublasGemmEx`)

    This variant uses NVIDIA’s built-in cuBLAS library to multiply matrices with 16-bit inputs and 32-bit accumulation. Under the hood it automatically picks up the GPU’s Tensor Cores for extra speed, and since it’s      part of cuBLAS it’s already highly tuned and ready for production.
  
  - ##### Custom CUDA Kernel (`cudaMatrixMultiply`)
  
    Here you write your own GPU kernel that works entirely in 32-bit floats. You decide exactly how threads and blocks are arranged, giving you full control over parallelism. It doesn’t rely on Tensor Cores, just         standard GPU math units.
  
  - ##### CPU SIMD (`simdMulOpt`)
  
    This version runs on the CPU and uses vector instructions (like SSE or AVX) to process multiple numbers at once. It doesn’t need a GPU at all, so it’s useful when you only have a regular processor. You can tweak      loop order and unrolling to match your memory system.
  
  - ##### WMMA Tensor-Core API (`wmmaMatrixMultiply`)
  
    With WMMA you work directly at the warp level to fire Tensor-Core instructions. You pick tile sizes for threadblocks, warps, and the tiny instruction tiles, then double-buffer loads to overlap data movement with      compute. This gives you hands-on control to squeeze out every last bit of Tensor-Core performance.  

### Testing algorithms runtime:

We added a Google Benchmark suite that runs each matrix-multiply variant end-to-end: it loads the same input matrices, dispatches them to the different implementations (**cuBLAS half-precision GEMM**, **custom CUDA kernel**, **CPU SIMD**, **WMMA**), times them under identical conditions (including device synchronization and memory clobber), and then reports the milliseconds per run so you can directly compare throughput and efficiency across all approaches.


### Restoring Accuracy:

Before restoring accuracy on Tensor cores, we tried to understand the logic by restot=ring it on CPU:

- ##### CPU Restore-Precision (`multiply_with_restored_precision`)

  On the CPU we split each double-precision value into a “high” float and a tiny “low” float remainder, then do four standard float-only matrix multiplies (high×high, low×high, high×low, low×low). Finally we add        those four results back in double precision. This trick recovers most of the lost bits without needing GPU hardware, but it runs in pure C++ loops and does all the work on your processor cores.

- ##### GPU WMMA Restore-Precision (`wmmaRestore`)

  On the GPU we load each element as a half-precision value plus a scaled residual also stored in half. We tile the work into 16×16 blocks and use NVIDIA’s WMMA API so each warp fires Tensor-Core MMA instructions on    both the main halves and the residual halves in one kernel. After accumulating the main and residual products (and undoing the scale), we get a full-precision result back on the device. This approach overlaps data    movement and compute in shared memory and squeezes out every Tensor-Core cycle for the extra accuracy.

### Prerequisites

#### Hardware
- **NVIDIA GPU** with Tensor Core support (compute capability ≥ 7.0, e.g. Volta/Turing/Ampere).  
- **CPU** with AVX2 support and at least 4 physical cores.  
- **Host RAM** ≥ 16 GB.

#### Compiler & Build Tools
- **GCC** ≥ 9.0 or **Clang** ≥ 10.0 with C++17 support.  
- **CMake** ≥ 3.18.  
- **Make** (Unix Makefiles).

#### CUDA Toolchain
- **CUDA Toolkit** ≥ 11.0, including:
  - `nvcc` compiler  
  - `cuBLAS` library  
  - CUDA headers (`cuda_runtime.h`, etc.)  
- NVIDIA driver compatible with your CUDA Toolkit (e.g. driver 450.x+ for CUDA 11).

#### CPU-side Libraries
- **Eigen** ≥ 3.3 for matrix baselines.  
- **OpenMP** (bundled with your compiler).

#### Benchmarks
- In order to run executables with benchmark you will need to install google benchmarks
  
### Building the Project

Once you’ve satisfied the prerequisites, follow these steps in your repo root:

1. **Create & enter a build directory**  
   ```bash
   mkdir -p build
   cd build
   ```
   
2. **Build directory**
   ```
   cmake -DCMAKE_BUILD_TYPE=Release ..
   cmake --build --parallel=8 .
   ```
3. **Run executables**
   ```
   ./name_of_exec <filepath>
   ```
