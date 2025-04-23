#include <vector>
#include <immintrin.h>
#include <cstring>
#include <cstdlib>
#include "matrixParser.h"

#define BLOCK_SIZE 64

void simdMulOpt(const float* A, const float* B, float* C,
                                  const size_t m, const size_t n, const size_t k) {
#pragma omp parallel for collapse(2)
    for (size_t i = 0; i < m; i += BLOCK_SIZE)
        for (size_t j = 0; j < k; j += BLOCK_SIZE)
            for (size_t kk = 0; kk < n; kk += BLOCK_SIZE)
                for (size_t ii = i; ii < std::min(i + BLOCK_SIZE, m); ii++) {
                    for (size_t j_block = j; j_block < std::min(j + BLOCK_SIZE, k); ) {
                        if (j_block + 8 <= std::min(j + BLOCK_SIZE, k)) {
                            __m256 sum = _mm256_setzero_ps();
                            for (size_t kk_iter = kk; kk_iter < std::min(kk + BLOCK_SIZE, n); kk_iter++) {
                                __m256 a = _mm256_set1_ps(A[ii * n + kk_iter]);
                                __m256 b = _mm256_loadu_ps(&B[kk_iter * k + j_block]);
                                sum = _mm256_fmadd_ps(a, b, sum);
                            }
                            const __m256 c_val = _mm256_loadu_ps(&C[ii * k + j_block]);
                            _mm256_storeu_ps(&C[ii * k + j_block], _mm256_add_ps(c_val, sum));
                            j_block += 8;
                        } else {
                            for (; j_block < std::min(j + BLOCK_SIZE, k); j_block++) {
                                float sum = 0.0f;
                                for (size_t kk_iter = kk; kk_iter < std::min(kk + BLOCK_SIZE, n); kk_iter++) {
                                    sum += A[ii * n + kk_iter] * B[kk_iter * k + j_block];
                                }
                                C[ii * k + j_block] += sum;
                            }
                        }
                    }
                }
}

int main(int argc, char** argv) {
    if(argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <matrix_file_path>" << std::endl;
        return 1;
    }
    const std::string filePath = argv[1];

//     size_t m, n, k;
//     parseDimensions(filePath, m, n, k);
//     const size_t A_elements = m * n;
//     const size_t B_elements = n * k;
//     const size_t C_elements = m * k;
//     auto* A = static_cast<float*>(malloc(A_elements * sizeof(float)));
//     auto* B = static_cast<float*>(malloc(B_elements * sizeof(float)));
//     auto* C = static_cast<float*>(malloc(C_elements * sizeof(float)));
//     std::vector<float> A, B;
//     loadMatrices_RR(filePath, A, B);

//     simdMulOpt(A, B, C, m, n, k);

//     free(A);
//     free(B);
//     free(C);
//     return 0;
}
