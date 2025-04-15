#include <iostream>
#include <cstdlib>
#include "matrixParser.h"
#include <cuda_runtime.h>

#define TILE_SIZE 16

__global__ void cudaMul(const float* A, const float* B, float* C, size_t m, size_t n, size_t k) {
    const size_t row = blockIdx.y * blockDim.y + threadIdx.y;
    const size_t col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < m && col < k) {
        float sum = 0.0f;
        for (size_t i = 0; i < n; i++) {
            sum += A[row * n + i] * B[i * k + col];
        }
        C[row * k + col] = sum;
    }
}
inline bool compareFloats(float a, float b, float epsilon = 1e-8) {
    return fabs(a - b) < epsilon;
}
void compare(const float* h_C, size_t m, size_t n, size_t k, const std::string& filePath) {
    size_t A_elements = m * n;
    size_t B_elements = n * k;

    // Allocate and reload host matrices A and B
    auto* A = static_cast<float*>(malloc(A_elements * sizeof(float)));
    auto* B = static_cast<float*>(malloc(B_elements * sizeof(float)));
    loadMatricesFromFileArray(filePath, A, A_elements, B, B_elements);

    // Allocate memory for CPU result
    auto* C_cpu = static_cast<float*>(malloc(m * k * sizeof(float)));

    // Brute-force matrix multiplication
    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < k; ++j) {
            float sum = 0.0f;
            for (size_t l = 0; l < n; ++l) {
                sum += A[i * n + l] * B[l * k + j];
            }
            C_cpu[i * k + j] = sum;
        }
    }

    // Compare CPU result with GPU result using epsilon for floating point comparison
    bool match = true;
    const float epsilon = 1e-5;
    for (size_t i = 0; i < m * k; ++i) {
        if (!compareFloats(C_cpu[i], h_C[i], epsilon)) {
            std::cerr << "Mismatch at index " << i << ": CPU = " << C_cpu[i]
                      << ", GPU = " << h_C[i] << std::endl;
            match = false;
            break;
        }
    }

    if (match) {
        std::cout << "Verification passed: CPU and GPU results match." << std::endl;
    } else {
        std::cout << "Verification failed: CPU and GPU results do not match." << std::endl;
    }

    free(A);
    free(B);
    free(C_cpu);
}

int main(const int argc, char** argv) {
    if(argc < 2) {
       std::cerr << "Usage: " << argv[0] << " <matrix_file_path>" << std::endl;
       return 1;
    }
    const std::string filePath = argv[1];

    size_t m, n, k;
    parseDimensions(filePath, m, n, k);
    const size_t A_elements = m * n, B_elements = n * k, C_elements = m * k;

    auto* h_A = static_cast<float*>(malloc(A_elements * sizeof(float)));
    auto* h_B = static_cast<float*>(malloc(B_elements * sizeof(float)));
    loadMatricesFromFileArray(filePath, h_A, A_elements, h_B, B_elements);

    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, A_elements * sizeof(float));
    cudaMalloc(&d_B, B_elements * sizeof(float));
    cudaMalloc(&d_C, C_elements * sizeof(float));

    cudaMemcpy(d_A, h_A, A_elements * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, B_elements * sizeof(float), cudaMemcpyHostToDevice);

    free(h_A);
    free(h_B);

    dim3 threadsPerBlock(TILE_SIZE, TILE_SIZE);
    dim3 blocksPerGrid((k + TILE_SIZE - 1) / TILE_SIZE, (m + TILE_SIZE - 1) / TILE_SIZE);

    cudaMul<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, m, n, k);
    cudaDeviceSynchronize();

    auto* h_C = static_cast<float*>(malloc(C_elements * sizeof(float)));
    cudaMemcpy(h_C, d_C, C_elements * sizeof(float), cudaMemcpyDeviceToHost);
    compare(h_C, m, n, k, filePath);
    std::cout << "CUDA multiplication complete." << std::endl;
    std::cout << "First element of result: " << h_C[0] << std::endl;

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_C);
    return 0;
}
