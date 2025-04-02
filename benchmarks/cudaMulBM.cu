#include <cstdlib>
#include "matrixParser.h"
#include <benchmark/benchmark.h>

#define TILE_SIZE 16

__global__ void matrixMultiplicationNaiveKernel(float *A, float *B, float *C, size_t m, size_t n, size_t k) {
    const size_t row = blockIdx.y * blockDim.y + threadIdx.y;
    const size_t col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < m && col < k) {
        float sum = 0.0f;
        for (int i = 0; i < n; i++) {
            sum += A[row * n + i] * B[i * k + col];
        }
        C[row * k + col] = sum;
    }
}

static void BM_cudaMul(benchmark::State &state, const std::string &filePath) {
    size_t m, n, k;
    parseDimensions(filePath, m, n, k);

    const size_t sizeA = m * n * sizeof(float);
    const size_t sizeB = n * k * sizeof(float);
    const size_t sizeC = m * k * sizeof(float);

    auto *h_A = static_cast<float*>(malloc(sizeA));
    auto *h_B = static_cast<float*>(malloc(sizeB));
    loadMatricesFromFileArray(filePath, h_A, m * n, h_B, n * k);

    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, sizeA);
    cudaMalloc(&d_B, sizeB);
    cudaMalloc(&d_C, sizeC);

    cudaMemcpy(d_A, h_A, sizeA, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, sizeB, cudaMemcpyHostToDevice);

    free(h_A);
    free(h_B);

    dim3 threadsPerBlock(TILE_SIZE, TILE_SIZE);
    dim3 blocksPerGrid((k + TILE_SIZE - 1) / TILE_SIZE,
                       (m + TILE_SIZE - 1) / TILE_SIZE);

    for (auto _ : state) {
        cudaMemset(d_C, 0, sizeC);
        matrixMultiplicationNaiveKernel<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, m, n, k);
        cudaDeviceSynchronize();
        benchmark::ClobberMemory();
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
