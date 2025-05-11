#include <cublas_v2.h>
#include "mmul.cuh"

void cublasMatrixMultiply(const float* h_A, const float* h_B, float* h_C,
                          const size_t m, const size_t k, const size_t n,
                          float& executionTime) {
    const size_t size_A = m * k * sizeof(float);
    const size_t size_B = k * n * sizeof(float);
    const size_t size_C = m * n * sizeof(float);

    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size_A);
    cudaMalloc(&d_B, size_B);
    cudaMalloc(&d_C, size_C);

    cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice);
    cudaMemset(d_C, 0, size_C);

    cublasHandle_t handle;
    cublasCreate(&handle);

    constexpr float alpha = 1.0f;
    constexpr float beta = 0.0f;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    cublasGemmEx(handle,
               CUBLAS_OP_N, CUBLAS_OP_N,
               n, m, k,
               &alpha,
               d_B, CUDA_R_32F, n,
               d_A, CUDA_R_32F, k,
               &beta,
               d_C, CUDA_R_32F, n,
               CUBLAS_COMPUTE_32F,
               CUBLAS_GEMM_DEFAULT);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&executionTime, start, stop);
    cudaDeviceSynchronize();
    cudaMemcpy(h_C, d_C, size_C, cudaMemcpyDeviceToHost);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cublasDestroy(handle);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

void cublasMatrixMultiplyHalf(const __half* h_A, const __half* h_B, float* h_C,
                          const size_t m, const size_t k, const size_t n,
                          float& executionTime) {
    const size_t size_A = m * k * sizeof(__half);
    const size_t size_B = k * n * sizeof(__half);
    const size_t size_C = m * n * sizeof(float);

    __half *d_A, *d_B;
    float *d_C;
    cudaMalloc(&d_A, size_A);
    cudaMalloc(&d_B, size_B);
    cudaMalloc(&d_C, size_C);

    cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice);
    cudaMemset(d_C, 0, size_C);

    cublasHandle_t handle;
    cublasCreate(&handle);

    constexpr float alpha = 1.0f;
    constexpr float beta = 0.0f;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    cublasGemmEx(handle,
               CUBLAS_OP_N, CUBLAS_OP_N,
               n, m, k,
               &alpha,
               d_B, CUDA_R_16F, n,
               d_A, CUDA_R_16F, k,
               &beta,
               d_C, CUDA_R_32F, n,
               CUBLAS_COMPUTE_32F,
               CUBLAS_GEMM_DEFAULT);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&executionTime, start, stop);

    cudaMemcpy(h_C, d_C, size_C, cudaMemcpyDeviceToHost);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cublasDestroy(handle);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}
