#include <iostream>
#include <vector>
#include <random>
#include <immintrin.h>
#include <iomanip>
#include <chrono>
#include <fstream>
#include <stdexcept>
#include "timeMeasurement.h"
#include "matrixParser.h"

std::vector<std::vector<float>> padMatrix(const std::vector<std::vector<float>>& matrix, int simdWidth = 8) {
    int originalRows = matrix.size();
    int originalCols = matrix[0].size();
    int paddedRows = (originalRows + simdWidth - 1) / simdWidth * simdWidth;
    int paddedCols = (originalCols + simdWidth - 1) / simdWidth * simdWidth;
    std::vector<std::vector<float>> paddedMatrix(paddedRows, std::vector<float>(paddedCols, 0.0f));
    for (int i = 0; i < originalRows; i++) {
        for (int j = 0; j < originalCols; j++) {
            paddedMatrix[i][j] = matrix[i][j];
        }
    }
    return paddedMatrix;
}

std::vector<std::vector<float>> removePadding(const std::vector<std::vector<float>>& paddedMatrix, int originalRows, int originalCols) {
    std::vector<std::vector<float>> originalMatrix(originalRows, std::vector<float>(originalCols, 0.0f));
    for (int i = 0; i < originalRows; i++) {
        for (int j = 0; j < originalCols; j++) {
            originalMatrix[i][j] = paddedMatrix[i][j];
        }
    }
    return originalMatrix;
}

std::vector<std::vector<float>> simdMatrixMultiply(const std::vector<std::vector<float>>& A, const std::vector<std::vector<float>>& B) {
    std::vector<std::vector<float>> paddedA = padMatrix(A);
    std::vector<std::vector<float>> paddedB = padMatrix(B);
    int N = paddedA.size();
    int M = paddedB.size();
    int P = paddedB[0].size();
    std::vector<std::vector<float>> C(N, std::vector<float>(P, 0.0f));
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < P; j += 8) {
            __m256 sum = _mm256_setzero_ps();
            for (int k = 0; k < M; k++) {
                __m256 a = _mm256_set1_ps(paddedA[i][k]);
                __m256 b = _mm256_loadu_ps(paddedB[k].data() + j);
                sum = _mm256_fmadd_ps(a, b, sum);
            }
            _mm256_storeu_ps(C[i].data() + j, sum);
        }
    }
    return C;
}

int main(int argc, char* argv[]) {
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <A_matrix_path> <B_matrix_path>" << std::endl;
        return 1;
    }

    std::string A_matrix_path = argv[1];
    std::string B_matrix_path = argv[2];

    std::vector<std::vector<float>> A = loadMatrixFromFile(A_matrix_path);
    std::vector<std::vector<float>> B = loadMatrixFromFile(B_matrix_path);

    if (A[0].size() != B.size()) {
        std::cerr << "Matrix dimensions do not match for multiplication." << std::endl;
        return 1;
    }

    auto start = get_current_time_fenced();
    std::vector<std::vector<float>> C = simdMatrixMultiply(A, B);
    auto end = get_current_time_fenced();

    std::cout << "Execution time: " << to_ms(end-start) << " ms" << std::endl;

    return 0;
}
