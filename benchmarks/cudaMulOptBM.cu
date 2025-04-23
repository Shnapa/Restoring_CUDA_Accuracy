#include <cuda_runtime.h>
#include <vector>
#include <cstdlib>
#include "../include/matrixParser.h"
#include <benchmark/benchmark.h>

#define TILE_SIZE 32

__global__ void matrixMultiplyTiled(const float *A, const float *B, float *C, const size_t m, const size_t n, const size_t k) {
    __shared__ float tileA[TILE_SIZE][TILE_SIZE];
    __shared__ float tileB[TILE_SIZE][TILE_SIZE];
    const size_t row = blockIdx.y * TILE_SIZE + threadIdx.y;
    const size_t col = blockIdx.x * TILE_SIZE + threadIdx.x;
    float sum = 0.0f;
    for (int t = 0; t < (n + TILE_SIZE - 1) / TILE_SIZE; t++) {
        if (row < m && t * TILE_SIZE + threadIdx.x < n)
            tileA[threadIdx.y][threadIdx.x] = A[row * n + t * TILE_SIZE + threadIdx.x];
        else
            tileA[threadIdx.y][threadIdx.x] = 0.0f;
        if (col < k && t * TILE_SIZE + threadIdx.y < n)
            tileB[threadIdx.y][threadIdx.x] = B[(t * TILE_SIZE + threadIdx.y) * k + col];
        else
            tileB[threadIdx.y][threadIdx.x] = 0.0f;
        __syncthreads();
        for (int i = 0; i < TILE_SIZE; i++)
            sum += tileA[threadIdx.y][i] * tileB[i][threadIdx.x];
        __syncthreads();
    }
    if (row < m && col < k)
        C[row * k + col] = sum;
}

static void BM_cudaMulOpt(benchmark::State& state, const std::string &filePath) {
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
    dim3 blocksPerGrid((k + TILE_SIZE - 1) / TILE_SIZE, (m + TILE_SIZE - 1) / TILE_SIZE);

    for (auto _ : state) {
        cudaMemset(d_C, 0, sizeC);
        matrixMultiplyTiled<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, m, n, k);
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
            BM_cudaMulOpt(state, filepath);
        });
    }
    benchmark::Initialize(&argc, argv);
    benchmark::RunSpecifiedBenchmarks();
    return 0;
}
