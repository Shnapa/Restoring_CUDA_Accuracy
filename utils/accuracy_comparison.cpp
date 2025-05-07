//
// Created by yuliana on 22.04.25.
//

#include "accuracy_comparison.h"
#include <vector>
#include <cmath>
#include <iostream>
#include <fstream>
#include <sstream>
#include <stdexcept>

double norm(const std::vector<float>& mat) {
    double sum = 0.0;
    for (float val : mat) {
        sum += static_cast<double>(val) * val;
    }
    return std::sqrt(sum);
}

double relativeResidual(const std::vector<float>& C_ref, const std::vector<float>& C_target) {
    if (C_ref.size() != C_target.size()) {
        throw std::runtime_error("Matrix sizes do not match");
    }

    std::vector<float> diff(C_ref.size());
    for (size_t i = 0; i < C_ref.size(); ++i) {
        diff[i] = C_ref[i] - C_target[i];
    }

    double norm_diff = norm(diff);
    double norm_ref = norm(C_ref);

    if (norm_ref == 0.0) return 0.0;

    return norm_diff / norm_ref;
}

std::vector<std::vector<float>> loadMatrix(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open file " + filename);
    }
    std::vector<std::vector<float>> matrix;
    std::string line;

    while (std::getline(file, line)) {
        std::istringstream stream(line);
        std::vector<float> row;
        float value;
        while (stream >> value) {
            row.push_back(value);
        }
        if (!row.empty()) matrix.push_back(row);
    }

    return matrix;
}

std::vector<std::vector<float>> multiplyNaive(const std::vector<std::vector<float>>& A,
                                               const std::vector<std::vector<float>>& B) {
    size_t m = A.size();
    size_t k = A[0].size();
    size_t n = B[0].size();

    std::vector<std::vector<float>> C(m, std::vector<float>(n, 0.0f));

    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < n; ++j) {
            for (size_t l = 0; l < k; ++l) {
                C[i][j] += A[i][l] * B[l][j];
            }
        }
    }

    return C;
}

float compareMatrices(const std::vector<std::vector<float>>& A,
                      const std::vector<std::vector<float>>& B) {
    if (A.size() != B.size() || A[0].size() != B[0].size()) {
        throw std::runtime_error("Matrix sizes do not match for comparison");
    }

    std::vector<float> flat_A, flat_B;
    for (size_t i = 0; i < A.size(); ++i) {
        for (size_t j = 0; j < A[0].size(); ++j) {
            flat_A.push_back(A[i][j]);
            flat_B.push_back(B[i][j]);
        }
    }

    return static_cast<float>(relativeResidual(flat_A, flat_B));
}
