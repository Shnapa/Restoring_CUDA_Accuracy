#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <iostream>
#include "matrixParser.h"
#include <benchmark/benchmark.h>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>

int loadHalfMatricesFromFileArray(const std::string &filePath, __half* A, size_t A_elements, __half* B, size_t B_elements) {
    std::ifstream file(filePath);
    if (!file.is_open()) {
        std::cerr << "Error: could not open file " << filePath << std::endl;
        return -1;
    }
    std::string line;
    if (!std::getline(file, line)) {
        std::cerr << "Error: could not read the first line from " << filePath << std::endl;
        return -1;
    }
    std::istringstream issA(line);
    size_t countA = 0;
    float value;
    while (issA >> value) {
        if (countA < A_elements) {
            A[countA++] = __float2half(value);
        } else {
            break;
        }
    }
    if (countA != A_elements) {
        std::cerr << "Error: Expected " << A_elements << " elements for Matrix A, but found " << countA << std::endl;
        return -1;
    }
    if (!std::getline(file, line)) {
        std::cerr << "Error: could not read the second line from " << filePath << std::endl;
        return -1;
    }
    std::istringstream issB(line);
    size_t countB = 0;
    while (issB >> value) {
        if (countB < B_elements) {
            B[countB++] = __float2half(value);
        } else {
            break;
        }
    }
    if (countB != B_elements) {
        std::cerr << "Error: Expected " << B_elements << " elements for Matrix B, but found " << countB << std::endl;
        return -1;
    }
    return 0;
}

static void BM_RunCublasMultiplication(benchmark::State& state, const std::string &filePath) {
    size_t m, n, k;
    parseDimensions(filePath, m, n, k);
    size_t A_elements = m * n;
    size_t B_elements = n * k;
    size_t C_elements = m * k;
    auto* h_A = static_cast<__half*>(malloc(A_elements * sizeof(__half)));
    auto* h_B = static_cast<__half*>(malloc(B_elements * sizeof(__half)));
    auto* h_C = static_cast<float*>(malloc(C_elements * sizeof(float)));

    if (loadHalfMatricesFromFileArray(filePath, h_A, A_elements, h_B, B_elements) != 0) {
        state.SkipWithError("Error loading matrices");
        return;
    }

    __half *d_A, *d_B;
    float *d_C;

    cudaMalloc(&d_A, A_elements * sizeof(__half));
    cudaMalloc(&d_B, B_elements * sizeof(__half));
    cudaMalloc(&d_C, C_elements * sizeof(float));

    cudaMemcpy(d_A, h_A, A_elements * sizeof(__half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, B_elements * sizeof(__half), cudaMemcpyHostToDevice);

    cublasHandle_t handle;
    cublasCreate(&handle);

    const float alpha = 1.0f;
    const float beta = 0.0f;

    for (auto _ : state) {
        cudaMemset(d_C, 0, C_elements * sizeof(float));
        cublasGemmEx(handle,
            CUBLAS_OP_N, CUBLAS_OP_N,
            m, k, n,
            &alpha,
            d_A, CUDA_R_16F, m,
            d_B, CUDA_R_16F, n,
            &beta,
            d_C, CUDA_R_32F, m,
            CUDA_R_32F,
            CUBLAS_GEMM_DEFAULT);
        cudaDeviceSynchronize();
    }

    cublasDestroy(handle);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);
}

int main(int argc, char** argv) {
    for (const auto &filepath : filePaths) {
        benchmark::RegisterBenchmark(filepath, [filepath](benchmark::State &state) {
            BM_RunCublasMultiplication(state, filepath);
        });
    }
    benchmark::Initialize(&argc, argv);
    benchmark::RunSpecifiedBenchmarks();
    return 0;
}