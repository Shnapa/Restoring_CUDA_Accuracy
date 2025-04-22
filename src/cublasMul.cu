#include <iostream>
#include <cstdlib>
#include "matrixParser.h"
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include "timeMeasurement.h"

int loadHalfMatricesFromFileArray(const std::string &filePath, __half* A, size_t A_elements, __half* B, size_t B_elements) {
    std::ifstream file(filePath);
    std::string line;
    std::getline(file, line);
    std::istringstream issA(line);
    size_t countA = 0;
    float value;
    while (issA >> value && countA < A_elements) {
        A[countA++] = __float2half(value);
    }
    std::getline(file, line);
    std::istringstream issB(line);
    size_t countB = 0;
    while (issB >> value && countB < B_elements) {
        B[countB++] = __float2half(value);
    }
    return 0;
}
inline bool compareFloats(float a, float b, float epsilon = 1e-7) {
    float res = std::abs((b - a)/a);
    return res < epsilon;
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
int main(int argc, char** argv) {
    if(argc < 2) {
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
    auto* h_C = static_cast<float*>(malloc(C_elements * sizeof(float)));

    loadMatricesFromFileArray(filePath, h_A, A_elements, h_B, B_elements);

    float *d_A, *d_B;
    float *d_C;

    cudaMalloc(&d_A, A_elements * sizeof(float));
    cudaMalloc(&d_B, B_elements * sizeof(float));
    cudaMalloc(&d_C, C_elements * sizeof(float));

    cudaMemcpy(d_A, h_A, A_elements * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, B_elements * sizeof(float), cudaMemcpyHostToDevice);

    free(h_A);
    free(h_B);

    cublasHandle_t handle;
    cublasCreate(&handle);

    constexpr float alpha = 1.0f;
    constexpr float beta = 0.0f;


    cudaMemset(d_C, 0, C_elements * sizeof(float));
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);
    cublasGemmEx(   handle,
                CUBLAS_OP_N, CUBLAS_OP_N,
                    m, k, n,
                    &alpha,
                    d_A, CUDA_R_32F, m,
                    d_B, CUDA_R_32F, n,
                    &beta,
                    d_C, CUDA_R_32F, m,
                    CUDA_R_32F,
                    CUBLAS_GEMM_DEFAULT);
    cudaDeviceSynchronize();

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    float elapsedTime = 0;
    cudaEventElapsedTime(&elapsedTime, start, stop);
    std::cout << "Time elapsed (ms): " << elapsedTime << std::endl;

    cudaMemcpy(h_C, d_C, C_elements * sizeof(float), cudaMemcpyDeviceToHost);
    compare(h_C, m, n, k, filePath);
    cublasDestroy(handle);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_C);
}
