#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <iostream>
#include <random>
#include "timeMeasurement.h"

int main(){
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(1.0f, 10000.0f);

    cublasHandle_t handle;
    cublasStatus_t status = cublasCreate(&handle);
    const int m = 1000, n = 1000, k = 1000;

    __half* h_A = static_cast<__half*>(malloc(m * k * sizeof(__half)));
    __half* h_B = static_cast<__half*>(malloc(k * n * sizeof(__half)));
    float*  h_C = static_cast<float*>(malloc(m * n * sizeof(float)));

    for (int i = 0; i < m*k; i++) {
        h_A[i] = __half2float(dist(gen));
        h_B[i] = __half2float(dist(gen));
        h_C[i] = 0.0f;
    }

    __half *d_A, *d_B;
    float *d_C;
    cudaMalloc((void**)&d_A, m * k * sizeof(__half));
    cudaMalloc((void**)&d_B, k * n * sizeof(__half));
    cudaMalloc((void**)&d_C, m * n * sizeof(float));

    cudaMemcpy(d_A, h_A, m * k * sizeof(__half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, k * n * sizeof(__half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, h_C, m * n * sizeof(float), cudaMemcpyHostToDevice);

    const float alpha = 1.0f;
    const float beta = 0.0f;

    // C = α × (A × B) + β × C
    status = cublasGemmEx(handle,
                         CUBLAS_OP_N, CUBLAS_OP_N, // transpose matrices or no, N - no
                         m, n, k,
                         &alpha,
                         d_A,
                         CUDA_R_16F, // FP16
                         m, // number of rows of A
                         d_B,
                         CUDA_R_16F, // FP16
                         k, // number of rows of B
                         &beta,
                         d_C,
                         CUDA_R_32F, // FP32(float)
                         m, // number of rows of C
                         CUDA_R_32F, // FP32(float)
                         CUBLAS_GEMM_DEFAULT);

    if (status != CUBLAS_STATUS_SUCCESS) {
        return -1;
    }

    cudaMemcpy(h_C, d_C, m * n * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cublasDestroy(handle);

}
