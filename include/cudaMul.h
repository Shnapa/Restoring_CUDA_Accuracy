//
// Created by yuliana on 01.04.25.
//

#ifndef CUDAMUL_H
#define CUDAMUL_H

#include <cuda_runtime.h>
#include <benchmark/benchmark.h>
#include <string>
#include <vector>

#define TILE_SIZE 16

__global__ void matrixMultiplicationNaiveKernel(float *A, float *B, float *C, size_t m, size_t n, size_t k);
static void BM_RunMultiplicationNaive(benchmark::State &state, const std::string &filePath);

#endif //CUDAMUL_H
