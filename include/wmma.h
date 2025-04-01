//
// Created by yuliana on 01.04.25.
//

#ifndef WMMA_H
#define WMMA_H

#include <cuda_fp16.h>
#include <mma.h>
#include <cuda_runtime.h>
#include <cstdlib>
#include <benchmark/benchmark.h>
#include <string>

using namespace nvcuda;

#define M 16
#define N 16
#define K 16

__global__ void matrixMultiplyWMMA(const half *A, const half *B, float *C, size_t m, size_t n, size_t k);

static void BM_RunMultiplicationWMMA(benchmark::State &state, const std::string &filePath);

#endif //WMMA_H
