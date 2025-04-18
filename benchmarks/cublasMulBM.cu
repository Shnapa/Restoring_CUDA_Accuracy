#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <iostream>
#include "matrixParser.h"
#include <benchmark/benchmark.h>
#include <fstream>
#include <vector>
#include <string>

int loadHalfMatricesFromFileArray(const std::string &filePath, __half* A, size_t A_elements, __half* B, size_t B_elements) {
    std::ifstream file(filePath);
    std::string line;
    std::getline(file, line);
    std::istringstream issA(line);
    size_t countA = 0;
    float value;
    while (issA >> value && countA < A_elements) {
        A[countA++] = __float2half(value);
    }
    std::getline(file, line);
    std::istringstream issB(line);
    size_t countB = 0;
    while (issB >> value && countB < B_elements) {
        B[countB++] = __float2half(value);
    }
    return 0;
}

static void BM_cublasMul(benchmark::State& state, const std::string &filePath) {
    size_t m, n, k;
    parseDimensions(filePath, m, n, k);
    const size_t A_elements = m * n;
    const size_t B_elements = n * k;
    const size_t C_elements = m * k;
    auto* h_A = static_cast<float*>(malloc(A_elements * sizeof(float)));
    auto* h_B = static_cast<float*>(malloc(B_elements * sizeof(float)));
    auto* h_C = static_cast<float*>(malloc(C_elements * sizeof(float)));

    loadMatricesFromFileArray(filePath, h_A, A_elements, h_B, B_elements);

    float *d_A, *d_B;
    float *d_C;

    cudaMalloc(&d_A, A_elements * sizeof(float));
    cudaMalloc(&d_B, B_elements * sizeof(float));
    cudaMalloc(&d_C, C_elements * sizeof(float));

    cudaMemcpy(d_A, h_A, A_elements * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, B_elements * sizeof(float), cudaMemcpyHostToDevice);

    free(h_A);
    free(h_B);

    cublasHandle_t handle;
    cublasCreate(&handle);

    constexpr float alpha = 1.0f;
    constexpr float beta = 0.0f;

    for (auto _ : state) {
        cudaMemset(d_C, 0, C_elements * sizeof(float));
        cublasGemmEx(handle,
            CUBLAS_OP_N, CUBLAS_OP_N,
            m, k, n,
            &alpha,
            d_A, CUDA_R_32F, m,
            d_B, CUDA_R_32F, n,
            &beta,
            d_C, CUDA_R_32F, m,
            CUDA_R_32F,
            CUBLAS_GEMM_DEFAULT);
        cudaDeviceSynchronize();
        cudaMemcpy(h_C, d_C, C_elements * sizeof(float), cudaMemcpyDeviceToHost);
        for (size_t i = 0; i < 10; i++) {
            std::cout << "C[" << i << "]: " << h_C[i] << "| ";
        }
    }

    cublasDestroy(handle);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    free(h_C);
}

int main(int argc, char** argv) {
    for (const auto &filepath : filePaths) {
        benchmark::RegisterBenchmark(filepath, [filepath](benchmark::State &state) {
            BM_cublasMul(state, filepath);
        });
    }
    benchmark::Initialize(&argc, argv);
    benchmark::RunSpecifiedBenchmarks();
    return 0;
}