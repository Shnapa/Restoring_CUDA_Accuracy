#include "accuracy_comparison.h"
#include <cmath>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <vector>

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
        float val;
        while (stream >> val) {
            row.push_back(val);
        }
        if (!row.empty()) matrix.push_back(row);
    }
    return matrix;
}

std::vector<std::vector<float>> multiplyNaive(const std::vector<std::vector<float>>& A,
                                              const std::vector<std::vector<float>>& B) {
    size_t m = A.size(), k = A[0].size(), n = B[0].size();
    std::vector<std::vector<float>> C(m, std::vector<float>(n, 0.0f));
    for (size_t i = 0; i < m; ++i)
        for (size_t j = 0; j < n; ++j)
            for (size_t l = 0; l < k; ++l)
                C[i][j] += A[i][l] * B[l][j];
    return C;
}

double norm(const std::vector<double>& v) {
    double sum = 0.0;
    for (double x : v) sum += x * x;
    return std::sqrt(sum);
}

double norm(const std::vector<float>& v) {
    double sum = 0.0;
    for (float x : v) sum += static_cast<double>(x) * x;
    return std::sqrt(sum);
}

double relativeResidual(const std::vector<double>& C_ref, const std::vector<float>& C_target) {
    if (C_ref.size() != C_target.size()) {
        throw std::runtime_error("Vector sizes do not match in relativeResidual()");
    }
    std::vector<double> diff(C_ref.size());
    for (size_t i = 0; i < C_ref.size(); ++i)
        diff[i] = C_ref[i] - static_cast<double>(C_target[i]);

    double norm_diff = norm(diff);
    double norm_ref = norm(C_ref);
    return (norm_ref == 0.0) ? 0.0 : norm_diff / norm_ref;
}

std::vector<double> referenceGEMM_FP64(const std::vector<float>& A, const std::vector<float>& B,
                                       size_t m, size_t k, size_t n) {
    std::vector<double> C(m * n, 0.0);
    for (size_t row = 0; row < m; ++row)
        for (size_t col = 0; col < n; ++col)
            for (size_t i = 0; i < k; ++i)
                C[row * n + col] += static_cast<double>(A[row * k + i]) * static_cast<double>(B[i * n + col]);
    return C;
}
