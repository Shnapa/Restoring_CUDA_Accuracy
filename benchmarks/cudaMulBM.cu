#include <cstdlib>
#include "../include/matrixParser.h"
#include <benchmark/benchmark.h>

#define TILE_SIZE 16

__global__ void matrixMultiplicationNaiveKernel(const float *A, const float *B, float *C, int m, int k, int n) {
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < m && col < k) {
        float sum = 0.0f;
        for (int i = 0; i < n; i++) {
            sum += A[row * n + i] * B[i * k + col];
        }
        C[row * k + col] = sum;
    }
}

static void BM_cudaMul(benchmark::State &state, const std::string &filePath) {
    int m, k, n;
    parseDimensions(filePath, m, k, n);

    const int size_A = m * k;
    const int size_B = k * n;
    const int size_C = m * n;

    std::vector<float> h_A(size_A), h_B(size_B), h_C(size_C);
    loadMatrices_RR(filePath, h_A, h_B);

    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size_A * sizeof(float));
    cudaMalloc(&d_B, size_B * sizeof(float));
    cudaMalloc(&d_C, size_C * sizeof(float));

    cudaMemcpy(d_A, h_A.data(), size_A * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B.data(), size_B * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(d_C, 0,        size_C * sizeof(float));

    dim3 threadsPerBlock(TILE_SIZE, TILE_SIZE);
    dim3 blocksPerGrid((n + TILE_SIZE - 1) / TILE_SIZE,
                       (m + TILE_SIZE - 1) / TILE_SIZE);

    for (auto _ : state) {
        matrixMultiplicationNaiveKernel<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, m, k, n);
        cudaDeviceSynchronize();
    }

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

}

int main(int argc, char** argv) {
    for (const auto &filepath : filePaths) {
        benchmark::RegisterBenchmark(filepath, [filepath](benchmark::State &state) {
            BM_cudaMul(state, filepath);
        });
    }
    benchmark::Initialize(&argc, argv);
    benchmark::RunSpecifiedBenchmarks();
    return 0;
}
