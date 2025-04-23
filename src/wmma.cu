#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <mma.h>
#include <regex>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <cmath>
#include "compare.cu"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "matrixParser.h"

using namespace nvcuda;

#define WARP_SIZE 32
#define TILE_DIM 16

template <typename T>
void padMatrix(const std::vector<T>& src,
               std::vector<T>& dst,
               const int rows, const int cols,
               const int padded_rows, const int padded_cols,
               T zero = T(0))
{
    dst.assign(padded_rows * padded_cols, zero);
    for (int i = 0; i < rows; ++i)
        for (int j = 0; j < cols; ++j)
            dst[i * padded_cols + j] = src[i * cols + j];
}

__global__ void wmmaMul(const __half *A,
                        const __half *B,
                        const float *C,
                        float *D,
                        const int padded_M, const int padded_N, const int padded_K)
{
    const int warpM = blockIdx.x;
    const int warpN = blockIdx.y;
    if (warpM * TILE_DIM >= padded_M || warpN * TILE_DIM >= padded_N)
        return;

    wmma::fragment<wmma::matrix_a, TILE_DIM, TILE_DIM, TILE_DIM, __half, wmma::row_major>   a_frag;
    wmma::fragment<wmma::matrix_b, TILE_DIM, TILE_DIM, TILE_DIM, __half, wmma::row_major>   b_frag;
    wmma::fragment<wmma::accumulator, TILE_DIM, TILE_DIM, TILE_DIM, float>               acc_frag;
    wmma::fragment<wmma::accumulator, TILE_DIM, TILE_DIM, TILE_DIM, float>               c_frag;

    wmma::fill_fragment(acc_frag, 0.0f);
    wmma::fill_fragment(c_frag, 0.0f);

    for (int tileK = 0; tileK < padded_K; tileK += TILE_DIM) {
        const half *tileA = A + warpM * TILE_DIM * padded_K + tileK;
        const half *tileB = B + tileK * padded_N + warpN * TILE_DIM;
        wmma::load_matrix_sync(a_frag, tileA, padded_K);
        wmma::load_matrix_sync(b_frag, tileB, padded_N);
        wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
    }

    const float *tileC = C + warpM * TILE_DIM * padded_N + warpN * TILE_DIM;
    wmma::load_matrix_sync(c_frag, tileC, padded_N, wmma::mem_row_major);

    for (int i = 0; i < nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, float, void>::num_elements; ++i)
        c_frag.x[i] += acc_frag.x[i];

    float *tileD = D + warpM * TILE_DIM * padded_N + warpN * TILE_DIM;
    wmma::store_matrix_sync(tileD, c_frag, padded_N, wmma::mem_row_major);
}

int main(const int argc, char* argv[])
{
    if (argc != 2) {
        fprintf(stderr, "Usage: %s matrix_MxKxN.txt\n", argv[0]);
        return EXIT_FAILURE;
    }
    const std::string filename = argv[1];

    int M, K, N;
    parseDimensions(filename, M, K, N);

    const int padded_M = (M + TILE_DIM - 1) / TILE_DIM * TILE_DIM;
    const int padded_K = (K + TILE_DIM - 1) / TILE_DIM * TILE_DIM;
    const int padded_N = (N + TILE_DIM - 1) / TILE_DIM * TILE_DIM;

    std::vector<float> A_f(M * K), B_f(K * N), C_f(M * N, 0.0f);
    loadMatrices_RR(filename, A_f, B_f);

    std::vector<__half> A_h(M * K), B_h(K * N);
    for (size_t i = 0; i < A_f.size(); ++i) A_h[i] = __float2half(A_f[i]);
    for (size_t i = 0; i < B_f.size(); ++i) B_h[i] = __float2half(B_f[i]);

    std::vector<__half>   A_pad, B_pad;
    std::vector<float> C_pad;
    padMatrix<__half>( A_h, A_pad, M, K, padded_M, padded_K, __float2half(0.0f));
    padMatrix<__half>( B_h, B_pad, K, N, padded_K, padded_N, __float2half(0.0f));
    padMatrix<float>(C_f, C_pad, M, N, padded_M, padded_N, 0.0f);

    __half  *d_A, *d_B;
    float *d_C, *d_D;
    const int sizeA = padded_M * padded_K;
    const int sizeB = padded_K * padded_N;
    const int sizeC = padded_M * padded_N;

    cudaMalloc(&d_A, sizeA * sizeof(__half));
    cudaMalloc(&d_B, sizeB * sizeof(__half));
    cudaMalloc(&d_C, sizeC * sizeof(float));
    cudaMalloc(&d_D, sizeC * sizeof(float));

    cudaMemcpy(d_A, A_pad.data(), sizeA * sizeof(__half),  cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B_pad.data(), sizeB * sizeof(__half),  cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, C_pad.data(), sizeC * sizeof(float), cudaMemcpyHostToDevice);

    dim3 threads(WARP_SIZE, 1);
    dim3 blocks(padded_M / TILE_DIM, padded_N / TILE_DIM);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    wmmaMul<<<blocks, threads>>>(d_A, d_B, d_C, d_D,
                                    padded_M, padded_N, padded_K);
    cudaDeviceSynchronize();
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);
    printf("GPU execution time: %.3f ms\n", ms);
    printf("TFLOPS: %.2f\n", (static_cast<double>(M) * N * K * 2) / (ms * 1e6));

    std::vector<float> D_h(sizeC);
    cudaMemcpy(D_h.data(), d_D, sizeC * sizeof(float), cudaMemcpyDeviceToHost);

    compare(D_h, M, K, N, filename);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaFree(d_D);
    return 0;
}
