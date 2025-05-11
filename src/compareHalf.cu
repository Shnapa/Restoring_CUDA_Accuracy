#include <cuda_fp16.h>
#include <iostream>
#include <vector>
#include <iomanip>
#include "matrixParser.h"
#include "mmul.cuh"

void compare_half(const std::vector<float>& h_C,
                  const size_t m, const size_t k, const size_t n,
                  const std::string& filePath)
{
    const size_t size_A = m * k;
    const size_t size_B = k * n;
    const size_t size_C = m * n;

    std::vector<__half> A(size_A), B(size_B);
    std::vector<float> C(size_C);

    loadMatrices_RR_half(filePath, A, B);

    float time = 0.0f;
    cublasMatrixMultiply(A.data(), B.data(), C.data(), m, k, n, time);

    constexpr std::array<float,8> eps = {
        1e-5f, 1e-6f, 1e-7f, 1e-8f,
        1e-9f, 1e-10f, 1e-11f, 1e-12f
    };

    std::vector firstMismatch(eps.size(), size_C);

    for (size_t i = 0; i < size_C; ++i) {
        const float cpu = C[i];
        const float gpu = h_C[i];
        for (size_t j = 0; j < eps.size(); ++j) {
            if (firstMismatch[j] == size_C && !compareFloats(cpu, gpu, eps[j])) {
                firstMismatch[j] = i;
            }
        }
        bool all_failed = true;
        for (const size_t idx : firstMismatch) {
            if (idx == size_C) {
                all_failed = false;
                break;
            }
        }
        if (all_failed) break;
    }

    for (size_t j = 0; j < eps.size(); ++j) {
        const float e = eps[j];
        if (firstMismatch[j] == size_C) {
            std::cout << "PASS @ eps=" << std::scientific << e << "\n";
        } else {
            const size_t idx = firstMismatch[j];
            std::cerr << std::fixed
                      << "FAIL @ eps=" << e
                      << " at idx=" << idx
                      << ": CPU=" << C[idx]
                      << " GPU=" << h_C[idx] << "\n";
        }
    }
}
