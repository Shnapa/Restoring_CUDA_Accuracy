#include <cuda_fp16.h>
#include <mma.h>
#include <cuda_runtime.h>
#include <vector>
#include <cstdlib>
#include "../include/matrixParser.h"
#include <benchmark/benchmark.h>

using namespace nvcuda;

#define TILE_SIZE 16
#define WARP_SIZE 32

void loadMatrices_RR_half(const std::string &filePath,
                     std::vector<__half> &A,
                     std::vector<__half> &B)
{
    size_t m, k, n;
    parseDimensions(filePath, m, k, n);
    std::ifstream fin(filePath);
    if (!fin.is_open()) std::exit(EXIT_FAILURE);
    std::string line;
    std::getline(fin, line);
    std::istringstream isa(line);
    float v;
    size_t i = 0;
    while (isa >> v && i < A.size()) A[i++] = __float2half(v);
    std::getline(fin, line);
    isa.clear();
    isa.str(line);
    i = 0;
    while (isa >> v && i < B.size()) B[i++] = __float2half(v);
}

__global__ void matrixMultiplyWMMA(const half* A, const half* B, float* C,
const size_t padded_M, const size_t padded_N, const size_t padded_K) {
    const size_t warpM = blockIdx.x;
    const size_t warpN = blockIdx.y;

    if (warpM * TILE_SIZE >= padded_M || warpN * TILE_SIZE >= padded_N)
        return;

    nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, TILE_SIZE, TILE_SIZE, TILE_SIZE, __half, nvcuda::wmma::row_major> a_frag;
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, TILE_SIZE, TILE_SIZE, TILE_SIZE, __half, nvcuda::wmma::row_major> b_frag;
    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, TILE_SIZE, TILE_SIZE, TILE_SIZE, float> acc_frag;

    fill_fragment(acc_frag, 0.0f);

    for (size_t tileK = 0; tileK < padded_K; tileK += TILE_SIZE) {
        const half* tileA = A + warpM * TILE_SIZE * padded_K + tileK;
        const half* tileB = B + tileK * padded_N + warpN * TILE_SIZE;

        load_matrix_sync(a_frag, tileA, padded_K);
        load_matrix_sync(b_frag, tileB, padded_N);
        mma_sync(acc_frag, a_frag, b_frag, acc_frag);
    }

    float* tileC = C + warpM * TILE_SIZE * padded_N + warpN * TILE_SIZE;
    store_matrix_sync(tileC, acc_frag, padded_N, nvcuda::wmma::mem_row_major);
}

static void BM_RunMultiplicationWMMA(benchmark::State &state, const std::string &filePath) {
    size_t m, k, n;
    parseDimensions(filePath, m, k, n);

    std::vector<float> h_A(m*k), h_B(n*k), h_C(m*n, 0.0f);
    loadMatrices_RR(filePath, h_A, h_B);

    const size_t padded_M = (m + TILE_SIZE - 1) / TILE_SIZE * TILE_SIZE;
    const size_t padded_K = (k + TILE_SIZE - 1) / TILE_SIZE * TILE_SIZE;
    const size_t padded_N = (n + TILE_SIZE - 1) / TILE_SIZE * TILE_SIZE;

    const auto h_A_pad = new __half[padded_M * padded_K];
    const auto h_B_pad = new __half[padded_K * padded_N];
    const auto h_C_pad = new float[padded_M * padded_N];

    for (size_t i = 0; i < padded_M; i++) {
        for (size_t j = 0; j < padded_K; j++) {
            if (i < m && j < k) {
                h_A_pad[i * padded_K + j] = __float2half(h_A[i * k + j]);
            } else {
                h_A_pad[i * padded_K + j] = __float2half(0.0f);
            }
        }
    }

    for (size_t i = 0; i < padded_K; i++) {
        for (size_t j = 0; j < padded_N; j++) {
            if (i < k && j < n) {
                h_B_pad[i * padded_N + j] = __float2half(h_B[i * n + j]);
            } else {
                h_B_pad[i * padded_N + j] = __float2half(0.0f);
            }
        }
    }

    for (size_t i = 0; i < padded_M * padded_N; i++) {
        h_C_pad[i] = 0.0f;
    }

    __half *d_A, *d_B;
    float *d_C;
    cudaMalloc(&d_A, padded_M * padded_K * sizeof(__half));
    cudaMalloc(&d_B, padded_K * padded_N * sizeof(__half));
    cudaMalloc(&d_C, padded_M * padded_N * sizeof(float));

    cudaMemcpy(d_A, h_A_pad, padded_M * padded_K * sizeof(__half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B_pad, padded_K * padded_N * sizeof(__half), cudaMemcpyHostToDevice);
    cudaMemset(d_C, 0, padded_M * padded_N * sizeof(float));

    dim3 threads(WARP_SIZE, 1);
    dim3 blocks(padded_M / TILE_SIZE, padded_N / TILE_SIZE);

    for (auto _ : state) {
        cudaMemset(d_C, 0, padded_M * padded_N * sizeof(float));
        matrixMultiplyWMMA<<<blocks, threads>>>(d_A, d_B, d_C, padded_M, padded_N, padded_K);
        cudaDeviceSynchronize();
        benchmark::ClobberMemory();
    }

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    delete[] h_A_pad;
    delete[] h_B_pad;
    delete[] h_C_pad;
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
