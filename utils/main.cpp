#include "accuracy_comparison.h"
#include <iostream>
#include <string>

std::vector<double> referenceGEMM_FP64(const std::vector<float>& A, const std::vector<float>& B, size_t m, size_t k, size_t n) {
    std::vector<double> C(m * n, 0.0);
    for (size_t row = 0; row < m; ++row) {
        for (size_t col = 0; col < n; ++col) {
            double sum = 0.0;
            for (size_t i = 0; i < k; ++i) {
                sum += static_cast<double>(A[row * k + i]) * static_cast<double>(B[i * n + col]);
            }
            C[row * n + col] = sum;
        }
    }
    return C;
}

int main(int argc, char** argv) {
    if (argc != 4) {
        std::cerr << "Usage: " << argv[0] << " --type matrixA.txt matrixB.txt\n";
        std::cerr << "Example: " << argv[0] << " --simd A.txt B.txt\n";
        return 1;
    }

    std::string flag = argv[1];
    std::string matrixA_path = argv[2];
    std::string matrixB_path = argv[3];

    std::vector<std::vector<float>> A = loadMatrix(matrixA_path);
    std::vector<std::vector<float>> B = loadMatrix(matrixB_path);

    std::vector<std::vector<float>> result_tested;
    std::vector<std::vector<float>> result_naive = multiplyNaive(A, B);

    if (flag == "--simd") {
        result_tested = ...;
    } else if (flag == "--simd-opt") {
        result_tested = ...;
    } else if (flag == "--cuda") {
        result_tested = ...;
    } else if (flag == "--cuda-opt") {
        result_tested = ...;
    } else if (flag == "--wmma") {
        result_tested = ...;
    } else if (flag == "--naive") {
        result_tested = result_naive;
    } else {
        std::cerr << "Unknown multiplication type flag: " << flag << std::endl;
        return 1;
    }

    float error = compareMatrices(result_tested, result_naive);

    std::cout << "Comparison with naive result: error = " << error << std::endl;

    if (error > 1e-7) {
        std::cerr << "Too large error. Probably incorrect implementation." << std::endl;
        return 1;
    }

    return 0;
}
