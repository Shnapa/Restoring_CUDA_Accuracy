// compare.cu
#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>

#include "matrixParser.h"

inline bool compareFloats(const float a, const float b, const float epsilon) {
    const float res = std::abs((b - a)/a);
    return res < epsilon;
}

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
            }
            C_cpu[i * n + j] = sum;
        }
    }

    constexpr float epsilon1 = 1e-5f;
    constexpr float epsilon2 = 1e-6f;
    constexpr float epsilon3 = 1e-7f;
    bool match_e5 = true;
    for (size_t idx = 0; idx < size_C; ++idx) {
        if (!compareFloats(C_cpu[idx], h_C[idx], epsilon1)) {
            std::cerr << std::setprecision(10) << "Mismatch at idx " << idx
                      << ": CPU = " << C_cpu[idx]
                      << " GPU = " << h_C[idx] << "\n";
            match_e5 = false;
            break;
        }
    }

    bool match_e6 = true;
    for (size_t idx = 0; idx < size_C; ++idx) {
        if (!compareFloats(C_cpu[idx], h_C[idx], epsilon2)) {
            std::cerr << std::setprecision(10) << "Mismatch at idx " << idx
                      << ": CPU = " << C_cpu[idx]
                      << " GPU = " << h_C[idx] << "\n";
            match_e6 = false;
            break;
        }
    }

    bool match_e7 = true;
    for (size_t idx = 0; idx < size_C; ++idx) {
        if (!compareFloats(C_cpu[idx], h_C[idx], epsilon3)) {
            std::cerr << std::setprecision(10) << "Mismatch at idx " << idx
                      << ": CPU = " << C_cpu[idx]
                      << " GPU = " << h_C[idx] << "\n";
            match_e7 = false;
            break;
        }
    }


    if (match_e5) std::cout << "Verification passed on 1e-5f: CPU and GPU results match.\n";
    else std::cout << "Verification failed on 1e-5f.\n";

    if (match_e6) std::cout << "Verification passed on 1e-6f: CPU and GPU results match.\n";
    else std::cout << "Verification failed on 1e-6f\n";

    if (match_e7) std::cout << "Verification passed on 1e-7f: CPU and GPU results match.\n";
    else std::cout << "Verification failed on 1e-7f\n";

}
