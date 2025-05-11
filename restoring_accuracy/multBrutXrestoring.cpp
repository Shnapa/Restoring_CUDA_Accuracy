#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <iomanip>
#include <algorithm>
#include <cfenv>

using Matrix = std::vector<double>;
using Matrix1 = std::vector<float>;
const double SCALE = std::pow(2, 24);

Matrix1 multiply_standard(const Matrix1& A, size_t m1, size_t n1,
                          const Matrix1& B, size_t m2, size_t n2) {
    Matrix1 C(m1 * n2, 0.0f);
    for (size_t i = 0; i < m1; ++i) {
        for (size_t j = 0; j < n2; ++j) {
            for (size_t k = 0; k < n1; ++k) {
                C[i * n2 + j] += A[i * n1 + k] * B[k * n2 + j];

            }
        }
    }
    return C;
}

Matrix multiply_standard1(const Matrix& A, size_t m1, size_t n1,
                          const Matrix& B, size_t m2, size_t n2) {
    Matrix C(m1 * n2, 0.0);
    for (size_t i = 0; i < m1; ++i) {
        for (size_t j = 0; j < n2; ++j) {
            for (size_t k = 0; k < n1; ++k) {
                C[i * n2 + j] += A[i * n1 + k] * B[k * n2 + j];
            }
        }
    }
    return C;
}

void split_to_hi_lo(const Matrix& input, Matrix1& hi, Matrix1& lo) {
    for (size_t i = 0; i < input.size(); ++i) {
        hi[i] = static_cast<float>(input[i]);
        lo[i] = static_cast<float>(input[i] - static_cast<double>(hi[i]));
    }
}

Matrix multiply_with_restored_precision(const Matrix& A, size_t m1, size_t n1,
                                        const Matrix& B, size_t m2, size_t n2) {
    Matrix1 A_hi(m1 * n1), A_lo(m1 * n1);
    Matrix1 B_hi(m2 * n2), B_lo(m2 * n2);

    split_to_hi_lo(A, A_hi, A_lo);
    split_to_hi_lo(B, B_hi, B_lo);

    Matrix1 P0 = multiply_standard(A_hi, m1, n1, B_hi, m2, n2);
    Matrix1 P1 = multiply_standard(A_lo, m1, n1, B_hi, m2, n2);
    Matrix1 P2 = multiply_standard(A_hi, m1, n1, B_lo, m2, n2);
    Matrix1 P3 = multiply_standard(A_lo, m1, n1, B_lo, m2, n2);

    Matrix C(m1 * n2);
    for (size_t i = 0; i < C.size(); ++i) {
        C[i] = static_cast<double>(P0[i]) + static_cast<double>(P1[i]) +
               static_cast<double>(P2[i]) + static_cast<double>(P3[i]);
    }

    return C;
}

Matrix generate_random_matrix(size_t rows, size_t cols, double min_val, double max_val) {
    Matrix mat(rows * cols);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(min_val, max_val);
    for (size_t i = 0; i < mat.size(); ++i) {
        mat[i] = dis(gen);
    }
    return mat;
}

void print_matrix(const Matrix& mat, size_t rows, size_t cols) {
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            std::cout << std::fixed << std::setprecision(52) << mat[i * cols + j] << "\t";
        }
        std::cout << std::endl;
    }
}

void compare_results(const Matrix& C_standard, const Matrix& C_restored, size_t rows, size_t cols) {
    double max_abs_error = 0.0;
    double max_rel_error = 0.0;
    double total_abs_error = 0.0;
    size_t count = rows * cols;

    for (size_t i = 0; i < count; ++i) {
        double abs_error = std::abs(C_standard[i] - C_restored[i]);
        double rel_error = (C_standard[i] != 0.0) ? abs_error / std::abs(C_standard[i]) : 0.0;
        max_abs_error = std::max(max_abs_error, abs_error);
        max_rel_error = std::max(max_rel_error, rel_error);
        total_abs_error += abs_error;
    }

    double avg_abs_error = total_abs_error / count;

    std::cout << std::fixed << std::setprecision(17);
    std::cout << "Максимальна абс похибка: " << max_abs_error << std::endl;
    std::cout << "Середня абс похибка: " << avg_abs_error << std::endl;
    std::cout << "Максимальна віднос похибка: " << max_rel_error << std::endl;

}

int main() {
    size_t m1 = 3, n1 = 2;
    size_t m2 = 2, n2 = 3;

    Matrix A = generate_random_matrix(m1, n1, 0, 1);
    Matrix B = generate_random_matrix(m2, n2, 0, 1);

    std::cout << "Матриця A:" << std::endl;
    print_matrix(A, m1, n1);
    std::cout << "Матриця B:" << std::endl;
    print_matrix(B, m2, n2);

    Matrix C_standard = multiply_standard1(A, m1, n1, B, m2, n2);
    Matrix C_restored = multiply_with_restored_precision(A, m1, n1, B, m2, n2);

    std::cout << "\nРезультат стандартного множення:" << std::endl;
    print_matrix(C_standard, m1, n2);
    std::cout << "\nРезультат множення з відновленням точності:" << std::endl;
    print_matrix(C_restored, m1, n2);

    std::cout << "\nПорівняння результатів:" << std::endl;
    compare_results(C_standard, C_restored, m1, n2);

    return 0;
}
