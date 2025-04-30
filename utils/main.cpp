#include "accuracy_comparison.h"
#include "mmul.cuh"
#include <iostream>
#include <string>
#include <vector>
#include <stdexcept>

std::vector<double> referenceGEMM_FP64(const std::vector<float>& A, const std::vector<float>& B, size_t m, size_t k, size_t n) {
    std::vector<double> C(m * n, 0.0);
    for (size_t row = 0; row < m; ++row) {
        for (size_t col = 0; col < n; ++col) {
            double sum = 0.0;
            for (size_t i = 0; i < k; ++i) {
                sum += static_cast<double>(A[row * k + i]) * static_cast<double>(B[i * n + col]);
            }
            C[row * n + col] = sum;
        }
    }
    return C;
}

std::vector<std::vector<float>> flattenAndCallCuda(
    void (*cudaFunc)(const float*, const float*, float*, size_t, size_t, size_t, float&),
    const std::vector<std::vector<float>>& A,
    const std::vector<std::vector<float>>& B,
    size_t m, size_t k, size_t n
) {
    std::vector<float> flatA(m * k);
    std::vector<float> flatB(k * n);
    std::vector<float> flatC(m * n);

    for (size_t i = 0; i < m; ++i)
        for (size_t j = 0; j < k; ++j)
            flatA[i * k + j] = A[i][j];

    for (size_t i = 0; i < k; ++i)
        for (size_t j = 0; j < n; ++j)
            flatB[i * n + j] = B[i][j];

    float execTime = 0.0f;
    cudaFunc(flatA.data(), flatB.data(), flatC.data(), m, k, n, execTime);

    std::vector<std::vector<float>> C(m, std::vector<float>(n));
    for (size_t i = 0; i < m; ++i)
        for (size_t j = 0; j < n; ++j)
            C[i][j] = flatC[i * n + j];

    return C;
}

std::vector<std::vector<float>> simdWrapper(
    const std::vector<std::vector<float>>& A,
    const std::vector<std::vector<float>>& B,
    size_t m, size_t k, size_t n
) {
    std::vector<float> flatA(m * k), flatB(k * n), flatC(m * n);
    for (size_t i = 0; i < m; ++i)
        for (size_t j = 0; j < k; ++j)
            flatA[i * k + j] = A[i][j];

    for (size_t i = 0; i < k; ++i)
        for (size_t j = 0; j < n; ++j)
            flatB[i * n + j] = B[i][j];

    simdMulOpt(flatA.data(), flatB.data(), flatC.data(), m, n, k);

    std::vector<std::vector<float>> C(m, std::vector<float>(n));
    for (size_t i = 0; i < m; ++i)
        for (size_t j = 0; j < n; ++j)
            C[i][j] = flatC[i * n + j];

    return C;
}

int main(int argc, char** argv) {
    if (argc != 4) {
        std::cerr << "Usage: " << argv[0] << " --type matrixA.txt matrixB.txt\n";
        return 1;
    }

    std::string flag = argv[1];
    std::string matrixA_path = argv[2];
    std::string matrixB_path = argv[3];

    auto A = loadMatrix(matrixA_path);
    auto B = loadMatrix(matrixB_path);

    size_t m = A.size();
    size_t k = A[0].size();
    size_t n = B[0].size();

    auto result_naive = multiplyNaive(A, B);
    std::vector<std::vector<float>> result_tested;

    try {
        if (flag == "--naive") {
            result_tested = result_naive;
        } else if (flag == "--simd" || flag == "--simd-opt") {
            result_tested = simdWrapper(A, B, m, k, n);
        } else if (flag == "--cuda") {
            result_tested = flattenAndCallCuda(cudaMatrixMultiply, A, B, m, k, n);
        } else if (flag == "--cuda-opt") {
            result_tested = flattenAndCallCuda(cudaMatrixMultiplyOptimized, A, B, m, k, n);
        } else if (flag == "--cublas") {
            result_tested = flattenAndCallCuda(cublasMatrixMultiply, A, B, m, k, n);
        } else if (flag == "--wmma") {
            result_tested = flattenAndCallCuda(wmmaMatrixMultiply, A, B, m, k, n);
        } else {
            std::cerr << "Unknown multiplication type flag: " << flag << std::endl;
            return 1;
        }

        float error = compareMatrices(result_tested, result_naive);
        std::cout << "Comparison with naive result: error = " << error << std::endl;

        if (error > 1e-7) {
            std::cerr << "Too large error. Probably incorrect implementation." << std::endl;
            return 1;
        }

    } catch (const std::exception& e) {
        std::cerr << "Error during multiplication: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
