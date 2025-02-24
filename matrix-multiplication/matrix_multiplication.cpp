//
// Created by hko on 24.02.2025.
//
#include "matrix_multiplication.h"

Matrix multiplication(const Matrix& A, const Matrix& B) {
        // розміри матриці
        size_t m1 = A.size();
        size_t n1 = A[0].size();
        size_t m2 = B.size();
        size_t n2 = B[0].size();
        // перевіряємо чи множення можливе
        if (n1 != m2) {
                throw std::invalid_argument("Matrix multiplication not possible: column count of A must match row count of B.");
        }
        Matrix result(m1, std::vector<int>(n2, 0));
        // власне множення, рядок на стовпчик
        for (size_t i = 0; i < m1; ++i) {
                for (size_t j = 0; j < n2; ++j) {
                        for (size_t k = 0; k < n1; ++k) {
                                result[i][j] += A[i][k] * B[k][j];
                        }
                }
        }
        return result;
}
void printing(const Matrix& C) {
        for (const auto& row : C) {
                for (int val : row) {
                        std::cout << val << " ";
                }
                std::cout << std::endl;
        }
}