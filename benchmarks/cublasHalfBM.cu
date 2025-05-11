#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <iostream>
#include "matrixParser.h"
#include "simdMM.h"
#include <benchmark/benchmark.h>
#include <fstream>
#include <vector>
#include <string>

void loadMatrices_RR_half(const std::string &filePath,
                     std::vector<__half> &A,
                     std::vector<__half> &B)
{
    size_t m, k, n;
    parseDimensions(filePath, m, k, n);
    std::ifstream fin(filePath);
    if (!fin.is_open()) std::exit(EXIT_FAILURE);
    std::string line;
    std::getline(fin, line);
    std::istringstream isa(line);
    float v;
    size_t i = 0;
    while (isa >> v && i < A.size()) A[i++] = __float2half(v);
    std::getline(fin, line);
    isa.clear();
    isa.str(line);
    i = 0;
    while (isa >> v && i < B.size()) B[i++] = __float2half(v);
}


static void BM_cublasMulHalf(benchmark::State& state, const std::string &filePath) {
    size_t m, k, n;
    parseDimensions(filePath, m, k, n);
    const size_t sizeA = m * k;
    const size_t sizeB = k * n;
    const size_t sizeC = m * n;

    std::vector<__half> h_A(sizeA), h_B(sizeB);
    std::vector<float> h_C(sizeC);
    loadMatrices_RR_half(filePath, h_A, h_B);

    __half *d_A, *d_B;
    float *d_C;
    cudaMalloc(&d_A, sizeA * sizeof(__half));
    cudaMalloc(&d_B, sizeB * sizeof(__half));
    cudaMalloc(&d_C, sizeC * sizeof(float));

    cudaMemcpy(d_A, h_A.data(), sizeA * sizeof(__half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B.data(), sizeB * sizeof(__half), cudaMemcpyHostToDevice);
    cudaMemset(d_C, 0, sizeC * sizeof(float));

    cublasHandle_t handle;
    cublasCreate(&handle);

    constexpr float alpha = 1.0f, beta = 0.0f;

    for (auto _ : state) {
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
        cudaDeviceSynchronize();
        benchmark::ClobberMemory();
    }
    cudaMemcpy(h_C.data(), d_C, sizeC * sizeof(float), cudaMemcpyDeviceToHost);

    cublasDestroy(handle);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

}

int main(int argc, char** argv) {
    for (const auto &filepath : filePaths) {
        benchmark::RegisterBenchmark(filepath, [filepath](benchmark::State &state) {
            BM_cublasMulHalf(state, filepath);
        });
    }
    benchmark::Initialize(&argc, argv);
    benchmark::RunSpecifiedBenchmarks();
    return 0;
}