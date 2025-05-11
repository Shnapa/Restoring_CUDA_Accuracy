#include "accuracy_comparison.h"
#include "mmul.h"
#include <iostream>
#include <string>
#include <vector>
#include <stdexcept>

std::vector<double> flattenMatrix64(const std::vector<std::vector<float>>& mat) {
    std::vector<double> result;
    for (const auto& row : mat) {
        for (float val : row) {
            result.push_back(static_cast<double>(val));
        }
    }
    return result;
}

int main(const int argc, char** argv) {
    if (argc != 4) {
        std::cerr << "Usage: " << argv[0] << " --type matrixA.txt matrixB.txt\n";
        return 1;
    }

    const std::string flag = argv[1];
    const std::string matrixA_path = argv[2];
    const std::string matrixB_path = argv[3];

    const auto A = loadMatrix(matrixA_path);
    const auto B = loadMatrix(matrixB_path);

    const size_t m = A.size();
    const size_t k = A[0].size();
    const size_t n = B[0].size();

    std::vector<std::vector<float>> result_tested;

    try {
        if (flag == "--naive") {
            result_tested = multiplyNaive(A, B);
        } else if (flag == "--simd" || flag == "--simd-opt") {
            result_tested = simdWrapper(A, B, m, k, n);
        } else if (flag == "--cuda") {
            result_tested = flattenAndCallCuda(cudaMatrixMultiply, A, B, m, k, n);
        } else if (flag == "--cuda-opt") {
            result_tested = flattenAndCallCuda(cudaMatrixMultiplyOptimized, A, B, m, k, n);
        } else if (flag == "--cublas") {
            result_tested = flattenAndCallCuda(cublasMatrixMultiply, A, B, m, k, n);
        } else if (flag == "--wmma") {
            result_tested = flattenAndCallCuda(wmmaMatrixMultiply, A, B, m, k, n);
        } else if (flag == "--restore_wmma") {
            result_tested = flattenAndCallCuda(wmmaRestore, A, B, m, k, n);
        } else {
            std::cerr << "Unknown multiplication type flag: " << flag << std::endl;
            return 1;
        }

        std::vector<float> flatA, flatB;
        for (const auto& row : A)
            flatA.insert(flatA.end(), row.begin(), row.end());
        for (const auto& row : B)
            flatB.insert(flatB.end(), row.begin(), row.end());

        std::vector<double> ref_fp64 = referenceGEMM_FP64(flatA, flatB, m, k, n);
        std::vector<double> flat_result_fp32;
        for (const auto& row : result_tested)
            for (float val : row)
                flat_result_fp32.push_back(static_cast<double>(val));

        const double residual = relativeResidual(ref_fp64, flat_result_fp32);

        std::cout << "RESIDUAL=" << residual << std::endl;

        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error during multiplication: " << e.what() << std::endl;
        return 1;
    }
}
