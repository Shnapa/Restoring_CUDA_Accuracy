//
// Created by hko on 24.02.2025.
//
#include <vector>
#include <iostream>
using Matrix = std::vector<std::vector<int>>;
Matrix multiplication(const Matrix& A, const Matrix& B) {
    size_t m1 = A.size();
    size_t n1 = A[0].size();
    size_t m2 = B.size();
    size_t n2 = B[0].size();
    Matrix result(m1, std::vector<int>(n2, 0));
    for (size_t i = 0; i < m1; ++i) {
        for (size_t j = 0; j < n2; ++j) {
            for (size_t k = 0; k < n1; ++k) {
                result[i][j] += A[i][k] * B[k][j];
            }
        }
    }
    return result;
}

