#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include "matrixParser.h"
#include "timeMeasurement.h"
#include "compare.cu"

int main(const int argc, char** argv) {
    if (argc < 2) return 1;
    const std::string filePath = argv[1];

    int m, k, n;
    parseDimensions(filePath, m, k, n);

    const int size_A = m * k;
    const int size_B = k * n;
    const int size_C = m * n;

    std::vector<float> h_A(size_A), h_B(size_B), h_C(size_C);
    loadMatrices_CC(filePath, h_A, h_B);

    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size_A * sizeof(float));
    cudaMalloc(&d_B, size_B * sizeof(float));
    cudaMalloc(&d_C, size_C * sizeof(float));

    cudaMemcpy(d_A, h_A.data(), size_A * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B.data(), size_B * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(d_C, 0, size_C * sizeof(float));

    cublasHandle_t handle;
    cublasCreate(&handle);

    constexpr float alpha = 1.0f, beta = 0.0f;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, nullptr);

    cublasGemmEx(handle,
                 CUBLAS_OP_T, CUBLAS_OP_T,
                 n, m, k,
                 &alpha,
                 d_B, CUDA_R_32F, k,
                 d_A, CUDA_R_32F, m,
                 &beta,
                 d_C, CUDA_R_32F, n,
                 CUDA_R_32F,
                 CUBLAS_GEMM_DEFAULT);

    cudaDeviceSynchronize();
    cudaEventRecord(stop, nullptr);
    cudaEventSynchronize(stop);

    float elapsed = 0.0f;
    cudaEventElapsedTime(&elapsed, start, stop);
    std::cout << elapsed << "\n";

    cudaMemcpy(h_C.data(), d_C, size_C * sizeof(float), cudaMemcpyDeviceToHost);

    compare(h_C, m, k, n, filePath);

    cublasDestroy(handle);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    return 0;
}