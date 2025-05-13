#include <iostream>
#include <vector>
#include <iomanip>
#include "compareMM.h"
#include "matrixParser.h"

void compare(const std::vector<float>& h_C,
             const size_t m, const size_t k, const size_t n,
             const std::string& filePath)
{
    const size_t size_A = m * k;
    const size_t size_B = k * n;
    const size_t size_C = m * n;

    std::vector<float> A(size_A), B(size_B), C_cpu(size_C);

    loadMatrices_RR(filePath, A, B);

    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < n; ++j) {
            float sum = 0.0f;
            for (size_t l = 0; l < k; ++l) {
                sum += A[i * k + l] * B[l * n + j];
                C_cpu[i * n + j] = sum;
            }
        }
    }

    constexpr std::array<float,8> eps = {
        1e-3f, 1e-4f, 1e-5f, 1e-6f, 1e-7f
    };

    std::vector<size_t> firstMismatch(eps.size(), size_C);

    for (size_t i = 0; i < size_C; ++i) {
        float cpu = C_cpu[i];
        float gpu = h_C[i];
        for (size_t j = 0; j < eps.size(); ++j) {
            if (firstMismatch[j] == size_C && !compareFloats(cpu, gpu, eps[j])) {
                firstMismatch[j] = i;
            }
        }
        bool all_failed = true;
        for (size_t idx : firstMismatch) {
            if (idx == size_C) {
                all_failed = false;
                break;
            }
        }
        if (all_failed) break;
    }

    for (size_t j = 0; j < eps.size(); ++j) {
        float e = eps[j];
        if (firstMismatch[j] == size_C) {
            std::cout << "PASS @ eps=" << std::scientific << e << "\n";
        } else {
            size_t idx = firstMismatch[j];
            std::cerr << std::fixed << std::setprecision(12)
                      << "FAIL @ eps=" << e
                      << " at idx=" << idx
                      << ": CPU=" << C_cpu[idx]
                      << " GPU=" << h_C[idx] << "\n";
        }
    }
}

