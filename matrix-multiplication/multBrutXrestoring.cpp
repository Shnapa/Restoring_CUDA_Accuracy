#include <iostream>
#include <vector>
#include <random>
#include <cmath>

using Matrix = std::vector<double>;
const double SCALE = 2048;


Matrix multiply_standard(const Matrix& A, size_t m1, size_t n1,
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

Matrix multiply_with_restored_precision(const Matrix& A, size_t m1, size_t n1,
                                       const Matrix& B, size_t m2, size_t n2) {
    Matrix A_hi(m1 * n1), A_lo(m1 * n1);
    Matrix B_hi(m2 * n2), B_lo(m2 * n2);

    for (size_t i = 0; i < A.size(); ++i) {
        A_hi[i] = static_cast<double>(static_cast<short>(A[i]));
        A_lo[i] = (A[i] - A_hi[i]) * SCALE;
    }
    for (size_t i = 0; i < B.size(); ++i) {
        B_hi[i] = static_cast<double>(static_cast<short>(B[i]));
        B_lo[i] = (B[i] - B_hi[i]) * SCALE;
    }

    Matrix P0 = multiply_standard(A_hi, m1, n1, B_hi, m2, n2);
    Matrix P1 = multiply_standard(A_lo, m1, n1, B_hi, m2, n2);
    Matrix P2 = multiply_standard(A_hi, m1, n1, B_lo, m2, n2);
    Matrix P3 = multiply_standard(A_lo, m1, n1, B_lo, m2, n2);

    Matrix C(m1 * n2);
    for (size_t i = 0; i < C.size(); ++i) {
        C[i] = P0[i] + (P1[i] + P2[i]) / SCALE + P3[i] / (SCALE * SCALE);
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
            std::cout << mat[i * cols + j] << "\t";
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

    std::cout << "Максимальна абсолютна похибка: " << max_abs_error << std::endl;
    std::cout << "Середня абсолютна похибка: " << avg_abs_error << std::endl;
    std::cout << "Максимальна відносна похибка: " << max_rel_error << std::endl;
}

int main() {
    size_t m1 = 3, n1 = 2;
    size_t m2 = 2, n2 = 3;

    Matrix A = generate_random_matrix(m1, n1, -10.0, 10.0);
    Matrix B = generate_random_matrix(m2, n2, -10.0, 10.0);

    std::cout << "Матриця A:" << std::endl;
    print_matrix(A, m1, n1);
    std::cout << "Матриця B:" << std::endl;
    print_matrix(B, m2, n2);

    Matrix C_standard = multiply_standard(A, m1, n1, B, m2, n2);
    Matrix C_restored = multiply_with_restored_precision(A, m1, n1, B, m2, n2);

    std::cout << "\nРезультат стандартного множення:" << std::endl;
    print_matrix(C_standard, m1, n2);
    std::cout << "\nРезультат множення з відновленням точності:" << std::endl;
    print_matrix(C_restored, m1, n2);

    std::cout << "\nПорівняння результатів (SCALE = 2048):" << std::endl;
    compare_results(C_standard, C_restored, m1, n2);

    return 0;
}