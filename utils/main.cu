#include "accuracy_comparison.h"
#include "../include/mmul.h"
#include <iostream>
#include <string>
#include <vector>
#include <stdexcept>

int main(int argc, char** argv) {
    if (argc != 4) {
        std::cerr << "Usage: " << argv[0] << " --type matrixA.txt matrixB.txt\n";
        return 1;
    }

    const std::string flag = argv[1];
    const std::string matrixA_path = argv[2];
    const std::string matrixB_path = argv[3];

    const auto A2D = loadMatrix(matrixA_path);
    const auto B2D = loadMatrix(matrixB_path);

    const size_t m = A2D.size();
    const size_t k = A2D[0].size();
    const size_t n = B2D[0].size();

    std::vector<float> flatA(m * k), flatB(k * n), flatC(m * n, 0.0f);
    for (size_t i = 0; i < m; ++i)
        for (size_t j = 0; j < k; ++j)
            flatA[i * k + j] = A2D[i][j];
    for (size_t i = 0; i < k; ++i)
        for (size_t j = 0; j < n; ++j)
            flatB[i * n + j] = B2D[i][j];

    try {
        if (flag == "--naive") {
            const auto result = multiplyNaive(A2D, B2D);
            flatC.clear();
            for (const auto& row : result)
                flatC.insert(flatC.end(), row.begin(), row.end());
        } else if (flag == "--simd" || flag == "--simd-opt") {
            simdMulOpt(flatA.data(), flatB.data(), flatC.data(), m, n, k);
        } else if (flag == "--cuda") {
            float execTime;
            cudaMatrixMultiply(flatA.data(), flatB.data(), flatC.data(), m, k, n, execTime);
        } else if (flag == "--cuda-opt") {
            float execTime;
            cudaMatrixMultiplyOptimized(flatA.data(), flatB.data(), flatC.data(), m, k, n, execTime);
        } else if (flag == "--cublas") {
            float execTime;
            cublasMatrixMultiply(flatA.data(), flatB.data(), flatC.data(), m, k, n, execTime);
        } else if (flag == "--wmma") {
            float execTime;
            wmmaMatrixMultiply(flatA.data(), flatB.data(), flatC.data(), m, k, n, execTime);
        } else if (flag == "--restore_wmma") {
            float execTime;
            wmmaRestore(flatA.data(), flatB.data(), flatC.data(), m, k, n, execTime);
        } else {
            std::cerr << "❌ Unknown multiplication type flag: " << flag << std::endl;
            return 1;
        }

        const std::vector<double> ref_fp64 = referenceGEMM_FP64(flatA, flatB, m, k, n);
        const double residual = relativeResidual(ref_fp64, flatC);

        std::cout << "RESIDUAL=" << residual << std::endl;
        return 0;

    } catch (const std::exception& e) {
        std::cerr << "❌ Error during multiplication: " << e.what() << std::endl;
        return 1;
    }
}
