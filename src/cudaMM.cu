#include "../include/mmul.h"
#include <cuda_runtime.h>

#define TILE_SIZE 16
#define WARP_SIZE 32

__global__ void cudaMulKernel(const float* A, const float* B, float* C,
                            const size_t m, const size_t k, const size_t n) {
    const size_t row = blockIdx.y * blockDim.y + threadIdx.y;
    const size_t col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < m && col < n) {
        float sum = 0.0f;
        for (size_t i = 0; i < k; ++i) {
            sum += A[row * k + i] * B[i * n + col];
        }
        C[row * n + col] = sum;
    }
}

__global__ void cudaMulOptKernel(const float* A, const float* B, float* C,
                               const size_t m, const size_t k, const size_t n) {
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

void cudaMatrixMultiply(const float* h_A, const float* h_B, float* h_C,
                      const size_t m, const size_t k, const size_t n,
                      float& executionTime) {
    const size_t size_A = m * k * sizeof(float);
    const size_t size_B = k * n * sizeof(float);
    const size_t size_C = m * n * sizeof(float);

    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size_A);
    cudaMalloc(&d_B, size_B);
    cudaMalloc(&d_C, size_C);

    cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice);
    cudaMemset(d_C, 0, size_C);

    dim3 threadsPerBlock(TILE_SIZE, TILE_SIZE);
    dim3 blocksPerGrid((n + TILE_SIZE - 1) / TILE_SIZE,
                     (m + TILE_SIZE - 1) / TILE_SIZE);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    cudaMulKernel<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, m, k, n);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    cudaEventElapsedTime(&executionTime, start, stop);

    cudaMemcpy(h_C, d_C, size_C, cudaMemcpyDeviceToHost);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

void cudaMatrixMultiplyOptimized(const float* h_A, const float* h_B, float* h_C,
                               const size_t m, const size_t k, const size_t n,
                               float& executionTime) {
    const size_t size_A = m * k * sizeof(float);
    const size_t size_B = k * n * sizeof(float);
    const size_t size_C = m * n * sizeof(float);

    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size_A);
    cudaMalloc(&d_B, size_B);
    cudaMalloc(&d_C, size_C);

    cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice);
    cudaMemset(d_C, 0, size_C);

    dim3 threadsPerBlock(TILE_SIZE, TILE_SIZE);
    dim3 blocksPerGrid((n + TILE_SIZE - 1) / TILE_SIZE,
                     (m + TILE_SIZE - 1) / TILE_SIZE);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    cudaMulOptKernel<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, m, k, n);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    cudaEventElapsedTime(&executionTime, start, stop);

    cudaMemcpy(h_C, d_C, size_C, cudaMemcpyDeviceToHost);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}