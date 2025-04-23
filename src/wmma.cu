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
void padMatrix(const std::vector<T>& src, std::vector<T>& dst, int rows, int cols, int padded_rows, int padded_cols, T zero = T(0)) {
    dst.resize(padded_rows * padded_cols, zero);
    for (int i = 0; i < rows; ++i)
        for (int j = 0; j < cols; ++j)
            dst[i * padded_cols + j] = src[i * cols + j];
}

void loadMatricesFromFile(const std::string& filename, std::vector<__half>& A, std::vector<__half>& B, std::vector<float>& C, int M, int K, int N) {
    std::ifstream infile(filename);
    if (!infile.is_open()) throw std::runtime_error("Could not open file");

    float val;
    int total_A = M * K;
    int total_B = K * N;
    int total_C = M * N;

    for (int i = 0; i < total_A; i++) {
        infile >> val;
        A.push_back(__float2half(val));
    }
    for (int i = 0; i < total_B; i++) {
        infile >> val;
        B.push_back(__float2half(val));
    }
    for (int i = 0; i < total_C; i++) {
        // infile >> val;
        C.push_back(0);
    }
}

__global__ void WMMAKernel(half *A, half *B, float *C, float *D, int M_GLOBAL, int N_GLOBAL, int K_GLOBAL, int padded_M, int padded_N, int padded_K) {
    int warpM = blockIdx.x;
    int warpN = blockIdx.y;

    if (warpM * TILE_DIM >= padded_M || warpN * TILE_DIM >= padded_N) return;

    wmma::fragment<wmma::matrix_a, TILE_DIM, TILE_DIM, TILE_DIM, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, TILE_DIM, TILE_DIM, TILE_DIM, half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, TILE_DIM, TILE_DIM, TILE_DIM, float> acc_frag;
    wmma::fragment<wmma::accumulator, TILE_DIM, TILE_DIM, TILE_DIM, float> c_frag;

    wmma::fill_fragment(acc_frag, 0.0f);
    wmma::fill_fragment(c_frag, 0.0f);

    for (int i = 0; i < padded_K; i += TILE_DIM) {
        const half *tileA = A + warpM * TILE_DIM * padded_K + i;
        const half *tileB = B + i * padded_N + warpN * TILE_DIM;
        wmma::load_matrix_sync(a_frag, tileA, padded_K);
        wmma::load_matrix_sync(b_frag, tileB, padded_N);
        wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
    }

    float *tileC = C + warpM * TILE_DIM * padded_N + warpN * TILE_DIM;

    for (int i = 0; i < acc_frag.num_elements; ++i)
        c_frag.x[i] += acc_frag.x[i];

    float *tileD = D + warpM * TILE_DIM * padded_N + warpN * TILE_DIM;
    wmma::store_matrix_sync(tileD, c_frag, padded_N, wmma::mem_row_major);
}

int main(const int argc, char* argv[]) {
    if (argc != 2) {
        printf("Usage: ./main matrix_MxKxN.txt\n");
        return 1;
    }

    const std::string filename = argv[1];
    size_t m, k, n;
    parseDimensions(filename, m, k, n);

    const int padded_M = ((m + TILE_DIM - 1) / TILE_DIM) * TILE_DIM;
    const int padded_K = ((k + TILE_DIM - 1) / TILE_DIM) * TILE_DIM;
    const int padded_N = ((n + TILE_DIM - 1) / TILE_DIM) * TILE_DIM;

    std::vector<__half> A, B;
    std::vector<float> C;
    loadMatricesFromFile(filename, A, B, C, m, k, n);

    std::vector<__half> A_padded, B_padded;
    std::vector<float> C_padded;
    padMatrix(A, A_padded, m, k, padded_M, padded_K, __float2half(0.0f));
    padMatrix(B, B_padded, k, n, padded_K, padded_N, __float2half(0.0f));
    padMatrix(C, C_padded, m, n, padded_M, padded_N, 0.0f);

    half *d_A, *d_B;
    float *d_C, *d_D;
    size_t size_A = padded_M * padded_K;
    size_t size_B = padded_K * padded_N;
    size_t size_C = padded_M * padded_N;

    cudaMalloc(&d_A, size_A * sizeof(half));
    cudaMalloc(&d_B, size_B * sizeof(half));
    cudaMalloc(&d_C, size_C * sizeof(float));
    cudaMalloc(&d_D, size_C * sizeof(float));

    cudaMemcpy(d_A, A_padded.data(), size_A * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B_padded.data(), size_B * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, C_padded.data(), size_C * sizeof(float), cudaMemcpyHostToDevice);

    dim3 threads(WARP_SIZE, 1);
    dim3 blocks(padded_M / TILE_DIM, padded_N / TILE_DIM);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    WMMAKernel<<<blocks, threads>>>(d_A, d_B, d_C, d_D, m, n, k, padded_M, padded_N, padded_K);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    printf("GPU execution time: %.3f ms\n", ms);
    printf("TFLOPS: %.2f\n", ((double)m * n * k * 2) / (ms * 1e6));

    std::vector<float> h_D(size_C);
    cudaMemcpy(h_D.data(), d_D, size_C * sizeof(float), cudaMemcpyDeviceToHost);

    printf("Result matrix D = A*B + C (%dx%d):\n", m, n);
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            printf("%.3f ", h_D[i * padded_N + j]);
        }
        printf("\n");
    }
    compare(h_D, m, k, n, filename);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaFree(d_D);
    return 0;
}
