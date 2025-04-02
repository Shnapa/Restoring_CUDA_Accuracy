#ifndef SIMD_MATRIX_MULTIPLY_H
#define SIMD_MATRIX_MULTIPLY_H

#include <vector>
#include <immintrin.h>
#include <cstring>
#include <cstdlib>
#include <string>
#include <benchmark/benchmark.h>

std::vector<float> padMatrix(const float* matrix, size_t rows, size_t cols, int simdWidth, size_t &paddedRows, size_t &paddedCols);

std::vector<float> simdMatrixMultiply(const float* A, const float* B, size_t A_rows, size_t A_cols, size_t B_cols);

static void BM_RunSimdMultiplication(benchmark::State &state, const std::string &filePath);

#endif
