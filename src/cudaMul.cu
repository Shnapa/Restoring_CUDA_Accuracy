#include <iostream>
#include <cstdlib>
#include "matrixParser.h"
#include <cuda_runtime.h>
#include "compare.cu"
#define TILE_SIZE 16

__global__ void cudaMul(const float* A, const float* B, float* C,
                        const size_t m, const size_t k, const size_t n) {
    const size_t row = blockIdx.y * blockDim.y + threadIdx.y;
    size_t col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < m && col < n) {
        float sum = 0.0f;
        for (size_t i = 0; i < k; ++i) {
            sum += A[row * k + i] * B[i * n + col];
        }
        C[row * n + col] = sum;
    }
}

int main(const int argc, char** argv) {
    if (argc < 2) {
        std::cerr << argv[0] << " <matrix_file_path>\n";
        return 1;
    }
    const std::string filePath = argv[1];

    size_t m, k, n;
    parseDimensions(filePath, m, k, n);

    const size_t size_A = m * k;
    const size_t size_B = k * n;
    const size_t size_C = m * n;

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

    cudaMul<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, m, k, n);
    cudaDeviceSynchronize();

    cudaMemcpy(h_C.data(), d_C, size_C * sizeof(float), cudaMemcpyDeviceToHost);

    compare(h_C, m, k, n, filePath, "rr");
    std::cout << "CUDA multiplication complete.\n"
              << "First element of result: " << h_C[0] << "\n";

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    return 0;
}
