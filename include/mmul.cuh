//
// Created by gllek-pc on 4/30/25.
//

#ifndef MMUL_H
#define MMUL_H
#pragma once

#include <vector>
#include <string>
#include <cuda_runtime.h>
#include <cublas_v2.h>

void cudaMatrixMultiply(const float* h_A, const float* h_B, float* h_C,
                        size_t m, size_t k, size_t n,
                        float& executionTime);

void cudaMatrixMultiplyOptimized(const float* h_A, const float* h_B, float* h_C,
                                size_t m, size_t k, size_t n,
                                float& executionTime);

void cublasMatrixMultiply(const float* h_A, const float* h_B, float* h_C,
                         size_t m, size_t k,  size_t n,
                         float& executionTime);

void wmmaMatrixMultiply(const float* h_A, const float* h_B, float* h_C,
                       size_t m, size_t k, size_t n,
                       float& executionTime);


void simdMulOpt(const float* A, const float* B, float* C,
               size_t m, size_t n, size_t k);
#endif //MMUL_H
