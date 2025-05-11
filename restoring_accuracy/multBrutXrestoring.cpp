#include <iostream>
#include <vector>
#include <random>
#include <tuple>
#include <cmath>
#include <iomanip>

using Matrix = std::vector<double>;
constexpr double SCALE = 2048;


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

float round_fp16(float x) {
    const int mantissa_bits = 10;
    float scale = static_cast<float>(1 << mantissa_bits);
    return std::round(x * scale) / scale;
}

Matrix multiply_with_restored_precision(const std::vector<double>& A, size_t m1, size_t n1,
                                        const std::vector<double>& B, size_t m2, size_t n2) {
    Matrix A_hi(m1 * n1), A_lo(m1 * n1);
    Matrix B_hi(m2 * n2), B_lo(m2 * n2);

    for (size_t i = 0; i < A.size(); ++i) {
        A_hi[i] = static_cast<float>(A[i]);
        A_lo[i] = static_cast<float>((A[i] - A_hi[i]) * SCALE);
    }

    for (size_t i = 0; i < B.size(); ++i) {
        B_hi[i] = static_cast<float>(B[i]);
        B_lo[i] = static_cast<float>((B[i] - B_hi[i]) * SCALE);
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

Matrix multiply_with_restored_precision_feng(const std::vector<double>& A, size_t m1, size_t n1,
                                            const std::vector<double>& B, size_t m2, size_t n2) {
    const double SPLIT_FACTOR = (1 << 27) + 1;
    double scale = 1 << 27;

    auto split = [](double x, double factor) -> std::pair<double, double> {
        double temp = factor * x;
        double x_hi = temp - (temp - x);
        double x_lo = x - x_hi;
        return {x_hi, x_lo};
    };

    Matrix A_hi(A.size()), A_lo(A.size());
    Matrix B_hi(B.size()), B_lo(B.size());

    for (size_t i = 0; i < A.size(); ++i) {
        std::tie(A_hi[i], A_lo[i]) = split(A[i], SPLIT_FACTOR);
    }

    for (size_t i = 0; i < B.size(); ++i) {
        std::tie(B_hi[i], B_lo[i]) = split(B[i], SPLIT_FACTOR);
    }

    Matrix P0 = multiply_standard(A_hi, m1, n1, B_hi, m2, n2);  // A_hi × B_hi
    Matrix P1 = multiply_standard(A_hi, m1, n1, B_lo, m2, n2);  // A_hi × B_lo
    Matrix P2 = multiply_standard(A_lo, m1, n1, B_hi, m2, n2);  // A_lo × B_hi
    Matrix P3 = multiply_standard(A_lo, m1, n1, B_lo, m2, n2);  // A_lo × B_lo

    Matrix C(m1 * n2);
    for (size_t i = 0; i < C.size(); ++i) {
        C[i] = P0[i] +
               (P1[i] + P2[i]) / scale +
               P3[i] / (scale * scale);
    }

    bool overflow_detected;
    do {
        overflow_detected = false;
        Matrix validation(C.size());

        for (size_t i = 0; i < C.size(); ++i) {
            validation[i] = C[i] * scale;
            if (std::isinf(validation[i])) {
                overflow_detected = true;
                scale /= 2.0;
                break;
            }
        }
    } while (overflow_detected);

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
            std::cout << std::fixed << std::setprecision(33) << mat[i * cols + j] << "\t";
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

    std::cout << "Maximum absolute error: " << max_abs_error << std::endl;
    std::cout << "Average absolute error: " << avg_abs_error << std::endl;
    std::cout << "Maximum relative error: " << max_rel_error << std::endl;

}

int main() {
    size_t m1 = 3, n1 = 2;
    size_t m2 = 2, n2 = 3;

    Matrix A = generate_random_matrix(m1, n1, -10.0, 10.0);
    Matrix B = generate_random_matrix(m2, n2, -10.0, 10.0);

    std::cout << "Matrix A:" << std::endl;
    print_matrix(A, m1, n1);
    std::cout << "Matrix B:" << std::endl;
    print_matrix(B, m2, n2);

    Matrix C_standard = multiply_standard(A, m1, n1, B, m2, n2);
    Matrix C_restored = multiply_with_restored_precision(A, m1, n1, B, m2, n2);
    Matrix C_restored_feng = multiply_with_restored_precision_feng(A, m1, n1, B, m2, n2);

    std::cout << "\nStandard multiplication result:" << std::endl;
    print_matrix(C_standard, m1, n2);
    std::cout << "\nMultiplication result with restored precision using Markidis method:" << std::endl;
    print_matrix(C_restored, m1, n2);
    std::cout << "\nMultiplication result with restored precision using Feng method:" << std::endl;
    print_matrix(C_restored_feng, m1, n2);

    std::cout << "\nComparison of results (Markidis) (SCALE = 2048):" << std::endl;
    compare_results(C_standard, C_restored, m1, n2);
    std::cout << "\nComparison of results (Feng) (SCALE = 2048):" << std::endl;
    compare_results(C_standard, C_restored_feng, m1, n2);

    return 0;
}