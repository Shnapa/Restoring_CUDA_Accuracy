#include <vector>
#include <immintrin.h>
#include <cstring>
#include <cstdlib>
#include "matrixParser.h"

std::vector<float> padMatrix(const float* matrix, size_t rows, size_t cols, int simdWidth, size_t &paddedRows, size_t &paddedCols) {
    paddedRows = ((rows + simdWidth - 1) / simdWidth) * simdWidth;
    paddedCols = ((cols + simdWidth - 1) / simdWidth) * simdWidth;
    std::vector<float> padded(paddedRows * paddedCols, 0.0f);
    for (size_t i = 0; i < rows; i++)
        for (size_t j = 0; j < cols; j++)
            padded[i * paddedCols + j] = matrix[i * cols + j];
    return padded;
}

std::vector<float> simdMul(const float* A, const float* B, size_t A_rows, size_t A_cols, size_t B_cols) {
    std::vector<float> C(A_rows * B_cols, 0.0f);
    for (size_t i = 0; i < A_rows; i++) {
        for (size_t j = 0; j < B_cols; j += 8) {
            __m256 sum = _mm256_setzero_ps();
            for (size_t k = 0; k < A_cols; k++) {
                __m256 a = _mm256_set1_ps(A[i * A_cols + k]);
                __m256 b = _mm256_loadu_ps(&B[k * B_cols + j]);
                sum = _mm256_fmadd_ps(a, b, sum);
            }
            _mm256_storeu_ps(&C[i * B_cols + j], sum);
        }
    }
    return C;
}

int main(int argc, char** argv) {
    if(argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <matrix_file_path>" << std::endl;
        return 1;
    }
    const std::string filePath = argv[1];

    size_t m, n, k;
    parseDimensions(filePath, m, n, k);
    const size_t A_elements = m * n;
    const size_t B_elements = n * k;
    auto* A_raw = static_cast<float*>(malloc(A_elements * sizeof(float)));
    auto* B_raw = static_cast<float*>(malloc(B_elements * sizeof(float)));
    std::vector<float> A, B;
    loadMatrices_RR(filePath, A, B);

    size_t paddedRowsA, paddedColsA;
    const std::vector<float> paddedA = padMatrix(A_raw, m, n, 8, paddedRowsA, paddedColsA);
    size_t paddedRowsB, paddedColsB;
    const std::vector<float> paddedB = padMatrix(B_raw, n, k, 8, paddedRowsB, paddedColsB);

    free(A_raw);
    free(B_raw);
    simdMul(paddedA.data(), paddedB.data(), paddedRowsA, paddedColsA, paddedColsB);
    return 0;
}
