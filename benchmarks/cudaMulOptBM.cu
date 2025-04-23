#include <cuda_runtime.h>
#include <vector>
#include <cstdlib>
#include "../include/matrixParser.h"
#include <benchmark/benchmark.h>

#define TILE_SIZE 32

__global__ void cudaMulOpt(const float* A, const float* B, float* C,
                           const int m, const int k, const int n)
{
    __shared__ float tileA[TILE_SIZE][TILE_SIZE];
    __shared__ float tileB[TILE_SIZE][TILE_SIZE];

    const size_t row = blockIdx.y * TILE_SIZE + threadIdx.y;
    const size_t col = blockIdx.x * TILE_SIZE + threadIdx.x;
    float sum = 0.0f;

    const size_t numTiles = (k + TILE_SIZE - 1) / TILE_SIZE;
    for (size_t t = 0; t < numTiles; ++t) {
        const size_t a_col = t * TILE_SIZE + threadIdx.x;
        const size_t b_row = t * TILE_SIZE + threadIdx.y;

        tileA[threadIdx.y][threadIdx.x] = (row < m && a_col < k)
            ? A[row * k + a_col]
            : 0.0f;
        tileB[threadIdx.y][threadIdx.x] = (b_row < k && col < n)
            ? B[b_row * n + col]
            : 0.0f;

        __syncthreads();
        for (int i = 0; i < TILE_SIZE; ++i) {
            sum += tileA[threadIdx.y][i] * tileB[i][threadIdx.x];
        }
        __syncthreads();
    }

    if (row < m && col < n) {
        C[row * n + col] = sum;
    }
}

static void BM_cudaMulOpt(benchmark::State& state, const std::string &filePath) {
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
        cudaMulOpt<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, m, n, k);
        cudaDeviceSynchronize();
    }

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

}

int main(int argc, char** argv) {
    for (const auto &filepath : filePaths) {
        benchmark::RegisterBenchmark(filepath, [filepath](benchmark::State &state) {
            BM_cudaMulOpt(state, filepath);
        });
    }
    benchmark::Initialize(&argc, argv);
    benchmark::RunSpecifiedBenchmarks();
    return 0;
}
