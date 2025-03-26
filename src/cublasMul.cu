#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <iostream>
#include <random>
#include "timeMeasurement.h"
#include "matrixParser.h"

int main(int argc, char* argv[]) {
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <A_matrix_path> <B_matrix_path> "<< std::endl;
        return 1;
    }

    std::string A_matrix_path = argv[1];
    std::string B_matrix_path = argv[2];

    int m, n, k;

    float* A = loadMatrixFromFileToArray(A_matrix_path, m, n);
    float* B = loadMatrixFromFileToArray(B_matrix_path, n, k);

    __half* h_A = static_cast<__half*>(malloc(m * n * sizeof(__half)));
    __half* h_B = static_cast<__half*>(malloc(n * k * sizeof(__half)));
    float* h_C = static_cast<float*>(malloc(m * k * sizeof(float)));

    for (int i = 0; i < m * n; i++) {
        h_A[i] = __float2half(A[i]);
    }

    for (int i = 0; i < n * k; i++) {
        h_B[i] = __float2half(B[i]);
    }

    __half *d_A, *d_B;
    float *d_C;
    size_t sizeA = m * n * sizeof(__half);
    size_t sizeB = n * k * sizeof(__half);
    size_t sizeC = m * k * sizeof(float);
    cudaMalloc(&d_A, sizeA);
    cudaMalloc(&d_B, sizeB);
    cudaMalloc(&d_C, sizeC);

    cudaMemcpy(d_A, h_A, sizeA, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, sizeB, cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, h_C, sizeC, cudaMemcpyHostToDevice);

    const float alpha = 1.0f;
    const float beta = 0.0f;

    cublasHandle_t handle;
    cublasCreate(&handle);

    auto start  = get_current_time_fenced();

    cublasStatus_t status = cublasGemmEx(
        handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        m, k, n,
        &alpha,
        d_A, CUDA_R_16F, m,
        d_B, CUDA_R_16F, n,
        &beta,
        d_C, CUDA_R_32F, m,
        CUDA_R_32F, CUBLAS_GEMM_DEFAULT
    );

    if (status != CUBLAS_STATUS_SUCCESS) {
        std::cerr << "CUBLAS gemm failed!" << std::endl;
        return -1;
    }

    auto end  = get_current_time_fenced();
    std::cout << "Elapsed time: " << to_ms(end-start) << " ms" << std::endl;

    cudaMemcpy(h_C, d_C, sizeC, cudaMemcpyDeviceToHost);

    cublasDestroy(handle);

    free(h_A);
    free(h_B);
    free(h_C);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}

