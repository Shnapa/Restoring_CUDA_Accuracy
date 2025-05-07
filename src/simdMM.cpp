#include <algorithm>
#include <immintrin.h>

#include "mmul.cuh"

#define BLOCK_SIZE 64

void simdMulOpt(const float* A, const float* B, float* C,
                const size_t m, const size_t n, const size_t k)
{
#pragma omp parallel for collapse(2)
    for (size_t i = 0; i < m; i += BLOCK_SIZE)
        for (size_t j = 0; j < k; j += BLOCK_SIZE)
            for (size_t kk = 0; kk < n; kk += BLOCK_SIZE)
                for (size_t ii = i; ii < std::min(i + BLOCK_SIZE, m); ii++) {
                    for (size_t j_block = j; j_block < std::min(j + BLOCK_SIZE, k); ) {
                        if (j_block + 8 <= std::min(j + BLOCK_SIZE, k)) {
                            __m256 sum = _mm256_setzero_ps();
                            for (size_t kk_iter = kk; kk_iter < std::min(kk + BLOCK_SIZE, n); kk_iter++) {
                                const __m256 a = _mm256_set1_ps(A[ii * n + kk_iter]);
                                const __m256 b = _mm256_loadu_ps(&B[kk_iter * k + j_block]);
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
