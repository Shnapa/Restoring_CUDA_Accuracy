

#include "matrixParser.h"
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <cstdlib>
#include <random>
#include <fstream>
#include <sstream>
#include "timeMeasurement.h"

#define TILE_SIZE 32

__global__ void matrixMultiplyTiled(float *A, float *B, float *C, int m, int n, int k) {
    __shared__ float tileA[TILE_SIZE][TILE_SIZE];
    __shared__ float tileB[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    float sum = 0.0f;
    for (int t = 0; t < (n + TILE_SIZE - 1) / TILE_SIZE; t++) {
        if (row < m && (t * TILE_SIZE + threadIdx.x) < n)
            tileA[threadIdx.y][threadIdx.x] = A[row * n + t * TILE_SIZE + threadIdx.x];
        else
            tileA[threadIdx.y][threadIdx.x] = 0.0f;
        if (col < k && (t * TILE_SIZE + threadIdx.y) < n)
            tileB[threadIdx.y][threadIdx.x] = B[(t * TILE_SIZE + threadIdx.y) * k + col];
        else
            tileB[threadIdx.y][threadIdx.x] = 0.0f;
        __syncthreads();
        for (int i = 0; i < TILE_SIZE; i++) {
            sum += tileA[threadIdx.y][i] * tileB[i][threadIdx.x];
        }
        __syncthreads();
    }
    if (row < m && col < k)
        C[row * k + col] = sum;
}


int main(int argc, char* argv[]) {
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <A_matrix_path> <B_matrix_path>" << std::endl;
        return 1;
    }

    std::string A_matrix_path = argv[1];
    std::string B_matrix_path = argv[2];

    int m, n, k;

    float* A = loadMatrixFromFileToArray(A_matrix_path, m, n);
    float* B = loadMatrixFromFileToArray(B_matrix_path, n, k);

    std::vector<float> h_A(A, A + m * n);
    std::vector<float> h_B(B, B + n * k);
    std::vector<float> h_C(m * k, 0.0f);

    float *d_A, *d_B, *d_C;
    size_t sizeA = m * n * sizeof(float);
    size_t sizeB = n * k * sizeof(float);
    size_t sizeC = m * k * sizeof(float);
    cudaMalloc(&d_A, sizeA);
    cudaMalloc(&d_B, sizeB);
    cudaMalloc(&d_C, sizeC);
    cudaMemcpy(d_A, h_A.data(), sizeA, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B.data(), sizeB, cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(TILE_SIZE, TILE_SIZE);
    dim3 blocksPerGrid((k + TILE_SIZE - 1) / TILE_SIZE, (m + TILE_SIZE - 1) / TILE_SIZE);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    matrixMultiplyTiled<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, m, n, k);

    cudaDeviceSynchronize();
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    float elapsed_ms = 0.0f;
    cudaEventElapsedTime(&elapsed_ms, start, stop);
    std::cout << "Elapsed time: " << elapsed_ms << " ms" << std::endl;

    cudaMemcpy(h_C.data(), d_C, sizeC, cudaMemcpyDeviceToHost);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}
