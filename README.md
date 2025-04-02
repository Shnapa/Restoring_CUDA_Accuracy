# Restoring the Accuracy of Calculations after Operations on CUDA Tensor Cores
### CUDA

We explored several methods for matrix multiplication using CUDA:

- **Default CUDA Multiplication**: A basic implementation using CUDA C++ kernels. This serves as a reference point for performance and accuracy, demonstrating how manual control over threads and memory can be used for matrix operations.

- **cuBLAS Multiplication**: Utilizes the `cublasGemmEx` function from the cuBLAS library to perform mixed-precision matrix multiplication. This approach automatically leverages Tensor Cores when possible, offering a highly optimized and production-grade solution.

- **WMMA (Warp Matrix Multiply-Accumulate)**: A low-level API that gives fine-grained control over Tensor Core operations. Using WMMA, we explicitly map thread warps to matrix tiles and invoke Tensor Core instructions directly, enabling experimentation with precision restoration techniques at the lowest level.

- **SIMD Multiplication**: To provide a performance and accuracy baseline, we implemented a SIMD (Single Instruction, Multiple Data) multiplication on CPU. While it is not CUDA-based, it helps highlight the trade-offs between CPU and GPU performance and numerical precision.

Due to the reduced precision formats (e.g., _FP16, TF32_) used by **Tensor Cores**, all GPU-based methods were evaluated not only for performance but also for numerical accuracy. This project focuses on mitigating precision loss by implementing:

- Error compensation strategies
- Improved rounding and accumulation outside Tensor Cores
- Scaling methods to prevent underflow in small values

**CUDA** plays a key role—not just for speed, but also for enabling detailed control over memory, precision, and thread execution patterns, which is crucial when developing and evaluating precision restoration techniques.
