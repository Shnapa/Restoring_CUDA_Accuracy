//
// Created by yuliana on 01.04.25.
//

#ifndef CUDAMULOPT_H
#define CUDAMULOPT_H

#include <cuda_runtime.h>
#include <cstdlib>
#include <benchmark/benchmark.h>

#define TILE_SIZE 32

__global__ void matrixMultiplyTiled(const float *A, const float *B, float *C, const size_t m, const size_t n, const size_t k);

static void BM_RunMultiplication(benchmark::State& state, const std::string &filePath);

#endif //CUDAMULOPT_H
