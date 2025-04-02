#include "multBrut.h"
#include <vector>
#include <iostream>

using Matrix = std::vector<int>;  // 1D matrix representation

Matrix multiplication(const Matrix& A, size_t m1, size_t n1, const Matrix& B, size_t m2, size_t n2) {
    Matrix result(m1 * n2, 0);  // 1D array to store the result

    for (size_t i = 0; i < m1; ++i) {
        for (size_t j = 0; j < n2; ++j) {
            for (size_t k = 0; k < n1; ++k) {
                result[i * n2 + j] += A[i * n1 + k] * B[k * n2 + j];
            }
        }
    }
    return result;
}

void printMatrix(const Matrix& mat, size_t rows, size_t cols) {
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            std::cout << mat[i * cols + j] << " ";
        }
        std::cout << std::endl;
    }
}
