//
// Created by gllek-pc on 4/23/25.
//
#include <string>
#include "../include/matrixParser.h"
#include "compare.cu"

int main(const int argc, char** argv) {
    if (argc < 2) return 1;
    const std::string filePath = argv[1];

    size_t m, k, n;
    parseDimensions(filePath, m, k, n);

    const size_t size_A = m * k;
    const size_t size_B = k * n;
    const size_t size_C = m * n;

    std::vector<float> h_A(size_A), h_B(size_B), h_C(size_C);
    loadMatrices_RR(filePath, h_A, h_B);

    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < n; ++j) {
            float sum = 0.0f;
            for (size_t l = 0; l < k; ++l) {
                sum += h_A[i * k + l] * h_B[l * n + j];
            }
            h_C[i * n + j] = sum;
        }
    }

    compare(h_C, m, k, n, filePath);

    for (size_t i = 0; i < 10; i++) {
        std::cout << h_C[i] << " ";
    }
}