#include "matrixParser.h"
#include "mmul.cuh"
#include "accuracy_comparison.h"

#include <iostream>
#include <string>
#include <vector>
#include <stdexcept>

int main(int argc, char** argv) {
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " --type matrixFile.txt\n";
        return 1;
    }

    const std::string flag = argv[1];
    const std::string filePath = argv[2];

    size_t m, k, n;
    parseDimensions(filePath, m, k, n);

    std::vector<float> A(m * k), B(k * n), C(m * n, 0.0f);

    // Load A and B from file (row-major format)
    loadMatrices_RR(filePath, A, B);

    try {
        if (flag == "--naive") {
            // Convert to 2D for naive
            std::vector<std::vector<float>> A2D(m, std::vector<float>(k));
            std::vector<std::vector<float>> B2D(k, std::vector<float>(n));
            for (size_t i = 0; i < m; ++i)
                for (size_t j = 0; j < k; ++j)
                    A2D[i][j] = A[i * k + j];
            for (size_t i = 0; i < k; ++i)
                for (size_t j = 0; j < n; ++j)
                    B2D[i][j] = B[i * n + j];

            const auto C2D = multiplyNaive(A2D, B2D);
            for (size_t i = 0; i < m; ++i)
                for (size_t j = 0; j < n; ++j)
                    C[i * n + j] = C2D[i][j];
        }
        else if (flag == "--simd" || flag == "--simd-opt") {
            simdMulOpt(A.data(), B.data(), C.data(), m, n, k);
        }
        else if (flag == "--cuda") {
            float execTime;
            cudaMatrixMultiply(A.data(), B.data(), C.data(), m, k, n, execTime);
        }
        else if (flag == "--cuda-opt") {
            float execTime;
            cudaMatrixMultiplyOptimized(A.data(), B.data(), C.data(), m, k, n, execTime);
        }
        else if (flag == "--cublas") {
            float execTime;
            cublasMatrixMultiply(A.data(), B.data(), C.data(), m, k, n, execTime);
        }
        else if (flag == "--wmma") {
            float execTime;
            wmmaMatrixMultiply(A.data(), B.data(), C.data(), m, k, n, execTime);
        }
        else if (flag == "--restore_wmma") {
            float execTime;
            wmmaRestore(A.data(), B.data(), C.data(), m, k, n, execTime);
        }
        else {
            std::cerr << "❌ Unknown multiplication type flag: " << flag << std::endl;
            return 1;
        }

        // Compute relative residual error
        std::vector<double> ref_fp64 = referenceGEMM_FP64(A, B, m, k, n);
        double residual = relativeResidual(ref_fp64, C);

        std::cout << "RESIDUAL=" << residual << std::endl;
        return 0;

    } catch (const std::exception& e) {
        std::cerr << "❌ Error: " << e.what() << std::endl;
        return 1;
    }
}
