//
// Created by hko on 25.03.2025.
//

#include "testing.h"
#include "matrix-multiplication/multBrut.h"
using Matrix = std::vector<int>;

std::tuple<Matrix, size_t, size_t> readMatrixFromFile(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file: " + filename);
    }

    size_t m, n;
    file >> m >> n;

    Matrix C(m * n);
    for (size_t i = 0; i < m * n; ++i) {
        file >> C[i];
    }

    return {C, m, n};
}

bool compareMatrices(const Matrix& A, size_t m1, size_t n1,
                     const Matrix& B, size_t m2, size_t n2,
                     const std::tuple<Matrix, size_t, size_t>& C_tuple) {
    const auto& [C, m3, n3] = C_tuple;

    if (m3 != m1 || n3 != n2) {
        return false;
    }

    Matrix correctMul = multiplication(A, m1, n1, B, m2, n2);

    if (C.size() != correctMul.size()) {
        return false;
    }

    for (size_t i = 0; i < C.size(); ++i) {
        if (C[i] != correctMul[i]) {
            return false;
        }
    }
    return true;
}
