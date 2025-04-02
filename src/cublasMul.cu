#include <iostream>
#include <cstdlib>
#include "matrixParser.h"
#include <cuda_runtime.h>
#include <cublas_v2.h>

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

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <matrix_file_path>" << std::endl;
        return 1;
    }
    const std::string filePath = argv[1];

    size_t m, n, k;
    parseDimensions(filePath, m, n, k);
    const size_t A_elements = m * n;
    const size_t B_elements = n * k;
    const size_t C_elements = m * k;

    auto* h_A = static_cast<float*>(malloc(A_elements * sizeof(float)));
    auto* h_B = static_cast<float*>(malloc(B_elements * sizeof(float)));
    loadMatricesFromFileArray(filePath, h_A, A_elements, h_B, B_elements);

    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, A_elements * sizeof(float));
    cudaMalloc(&d_B, B_elements * sizeof(float));
    cudaMalloc(&d_C, C_elements * sizeof(float));

    cudaMemcpy(d_A, h_A, A_elements * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, B_elements * sizeof(float), cudaMemcpyHostToDevice);

    cublasHandle_t handle;
    cublasCreate(&handle);
    constexpr float alpha = 1.0f, beta = 0.0f;
    cublasSgemm(handle,
                CUBLAS_OP_N, CUBLAS_OP_N,
                m, k, n,
                &alpha,
                d_A, m,
                d_B, n,
                &beta,
                d_C, m);
    cudaDeviceSynchronize();

    auto* h_C = static_cast<float*>(malloc(C_elements * sizeof(float)));
    cudaMemcpy(h_C, d_C, C_elements * sizeof(float), cudaMemcpyDeviceToHost);

    std::cout << "CUBLAS multiplication complete." << std::endl;
    std::cout << "First element of result: " << h_C[0] << std::endl;

    cublasDestroy(handle);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);
    return 0;
}
