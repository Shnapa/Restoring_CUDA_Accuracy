#include <cuda_fp16.h>
#include <mma.h>
#include <cuda_runtime.h>
#include <vector>
#include <cstdlib>
#include "../include/matrixParser.h"
#include <benchmark/benchmark.h>

using namespace nvcuda;

#define M 16
#define N 16
#define K 16
#define WMMA_N 32

int loadHalfMatricesFromFileArray(const std::string &filePath, __half* A, size_t A_elements, __half* B, size_t B_elements) {
    std::ifstream file(filePath);
    std::string line;
    std::getline(file, line);
    std::istringstream issA(line);
    size_t countA = 0;
    float value;
    while (issA >> value && countA < A_elements) {
        A[countA++] = __float2half(value);
    }
    std::getline(file, line);
    std::istringstream issB(line);
    size_t countB = 0;
    while (issB >> value && countB < B_elements) {
        B[countB++] = __float2half(value);
    }
    return 0;
}

__global__ void matrixMultiplyWMMA(const __half *A, const __half *B, float *C, size_t m, size_t n, size_t k) {
    const size_t warpM = (blockIdx.y * blockDim.y + threadIdx.y) * M;
    const size_t warpN = (blockIdx.x * blockDim.x + threadIdx.x) * N;

    if (warpM >= m || warpN >= n) return;

    wmma::fragment<wmma::matrix_a, M, N, K, __half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, M, N, K, __half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, M, N, K, float> acc_frag;

    wmma::fill_fragment(acc_frag, 0.0f);

    for (int i = 0; i < k; i += K) {
        wmma::load_matrix_sync(a_frag, A + warpM * k + i, k);
        wmma::load_matrix_sync(b_frag, B + i * n + warpN, n);
        wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
    }

    wmma::store_matrix_sync(C + warpM * n + warpN, acc_frag, n, wmma::mem_row_major);
}

static void BM_RunMultiplicationWMMA(benchmark::State &state, const std::string &filePath) {
    int m, n, k;
    parseDimensions(filePath, m, n, k);

    const int sizeA = m * k * sizeof(__half);
    const int sizeB = k * n * sizeof(__half);
    const int sizeC = m * n * sizeof(float);

    auto *h_A = static_cast<half*>(malloc(sizeA));
    auto *h_B = static_cast<half*>(malloc(sizeB));

    loadHalfMatricesFromFileArray(filePath, h_A, m * k, h_B, k * n);

    half *d_A, *d_B;
    float *d_C;
    cudaMalloc(&d_A, sizeA);
    cudaMalloc(&d_B, sizeB);
    cudaMalloc(&d_C, sizeC);

    cudaMemcpy(d_A, h_A, sizeA, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, sizeB, cudaMemcpyHostToDevice);

    free(h_A);
    free(h_B);

    dim3 threadsPerBlock(WMMA_N, WMMA_N);
    dim3 blocksPerGrid((n + N * threadsPerBlock.x - 1) / (N * threadsPerBlock.x),
                       (m + M * threadsPerBlock.y - 1) / (M * threadsPerBlock.y));

    for (auto _ : state) {
        cudaMemset(d_C, 0, sizeC);
        matrixMultiplyWMMA<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, m, n, k);
        cudaDeviceSynchronize();
        benchmark::ClobberMemory();
    }

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

int main(int argc, char** argv) {
    for (const auto &filepath : filePaths) {
        benchmark::RegisterBenchmark(filepath, [filepath](benchmark::State &state) {
            BM_RunMultiplicationWMMA(state, filepath);
        });
    }
    benchmark::Initialize(&argc, argv);
    benchmark::RunSpecifiedBenchmarks();
    return 0;
}
