#include <vector>
#include <immintrin.h>
#include <cstring>
#include <cstdlib>
#include "../include/matrixParser.h"
#include <benchmark/benchmark.h>

#define BLOCK_SIZE 64

void simdMatrixMultiply(const float* A, const float* B, float* C,
                                  const size_t m, const size_t k, const size_t n) {
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

static void BM_simdMulOpt(benchmark::State &state, const std::string &filePath) {
    size_t m, k, n;
    parseDimensions(filePath, m, k, n);

    const size_t size_A = m * k;
    const size_t size_B = k * n;
    const size_t size_C = m * n;

    std::vector<float> h_A(size_A), h_B(size_B), h_C(size_C);
    loadMatrices_RR(filePath, h_A, h_B);

    for (auto _ : state) {
        simdMatrixMultiply(h_A.data(), h_B.data(), h_C.data(), m, k, n);
        benchmark::ClobberMemory();
    }
}

int main(int argc, char** argv) {
    for (const auto & filepath : filePaths) {
        benchmark::RegisterBenchmark(filepath, [filepath](benchmark::State &state) {
            BM_simdMulOpt(state, filepath);
        });
    }
    benchmark::Initialize(&argc, argv);
    benchmark::RunSpecifiedBenchmarks();
    return 0;
}
