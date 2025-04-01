#ifndef SIMD_BLOCKED_MATRIX_MULTIPLY_H
#define SIMD_BLOCKED_MATRIX_MULTIPLY_H

#include <cstddef>
#include <string>
#include <benchmark/benchmark.h>

void simdMatrixMultiply(const float* A, const float* B, float* C,
                        const size_t m, const size_t n, const size_t k);

void BM_RunSimdMultiplication(benchmark::State &state, const std::string &filePath);

#endif
