#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <mma.h>
#include <regex>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

using namespace nvcuda;

#define WARP_SIZE 32
#define TILE_DIM 16

void parseDimensionsFromFilename(const std::string& filename, int& M, int& K, int& N) {
    std::regex pattern(".*_(\\d+)x(\\d+)x(\\d+)\\.txt");
    std::smatch match;
    if (std::regex_match(filename, match, pattern)) {
        M = std::stoi(match[1]);
        K = std::stoi(match[2]);
        N = std::stoi(match[3]);
    } else {
        throw std::runtime_error("Filename does not match expected format");
    }
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
        A[i] = __float2half(val);
    }
    for (int i = 0; i < total_B; i++) {
        infile >> val;
        B[i] = __float2half(val);
    }
    for (int i = 0; i < total_C; i++) {
        infile >> val;
        C[i] = val;
    }
}

__global__ void WMMAKernel(half *A, half *B, float *C, float *D, int M_GLOBAL, int N_GLOBAL, int K_GLOBAL) {
    int warpM = blockIdx.x;
    int warpN = blockIdx.y;

    if ((warpM + 1) * TILE_DIM > M_GLOBAL || (warpN + 1) * TILE_DIM > N_GLOBAL)
        return;

    wmma::fragment<wmma::matrix_a, TILE_DIM, TILE_DIM, TILE_DIM, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, TILE_DIM, TILE_DIM, TILE_DIM, half, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, TILE_DIM, TILE_DIM, TILE_DIM, float> acc_frag;
    wmma::fragment<wmma::accumulator, TILE_DIM, TILE_DIM, TILE_DIM, float> c_frag;

    wmma::fill_fragment(acc_frag, 0.0f);

    for (int i = 0; i < K_GLOBAL; i += TILE_DIM) {
        if ((i + TILE_DIM) <= K_GLOBAL) {
            const half *tileA = A + warpM * TILE_DIM * K_GLOBAL + i;
            const half *tileB = B + i * N_GLOBAL + warpN * TILE_DIM;
            wmma::load_matrix_sync(a_frag, tileA, K_GLOBAL);
            wmma::load_matrix_sync(b_frag, tileB, N_GLOBAL);
            wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
        }
    }

    float *tileC = C + warpM * TILE_DIM * N_GLOBAL + warpN * TILE_DIM;
    wmma::load_matrix_sync(c_frag, tileC, N_GLOBAL, wmma::mem_row_major);

    for (int i = 0; i < acc_frag.num_elements; ++i)
        c_frag.x[i] += acc_frag.x[i];

    float *tileD = D + warpM * TILE_DIM * N_GLOBAL + warpN * TILE_DIM;
    wmma::store_matrix_sync(tileD, c_frag, N_GLOBAL, wmma::mem_row_major);
}

int main(int argc, char* argv[]) {
    if (argc != 2) {
        printf("Usage: ./main matrix_MxKxN.txt\n");
        return 1;
    }

    std::string filename = argv[1];
    int M, K, N;
    parseDimensionsFromFilename(filename, M, K, N);

    if (M < TILE_DIM || K < TILE_DIM || N < TILE_DIM) {
        printf("Matrix dimensions must be at least 16x16x16 for WMMA\n");
        return 0;
    }

    size_t size_A = M * K;
    size_t size_B = K * N;
    size_t size_C = M * N;

    std::vector<__half> h_A(size_A);
    std::vector<__half> h_B(size_B);
    std::vector<float> h_C(size_C);

    loadMatricesFromFile(filename, h_A, h_B, h_C, M, K, N);

    half *d_A, *d_B;
    float *d_C, *d_D;
    cudaMalloc(&d_A, size_A * sizeof(half));
    cudaMalloc(&d_B, size_B * sizeof(half));
    cudaMalloc(&d_C, size_C * sizeof(float));
    cudaMalloc(&d_D, size_C * sizeof(float));

    cudaMemcpy(d_A, h_A.data(), size_A * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B.data(), size_B * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, h_C.data(), size_C * sizeof(float), cudaMemcpyHostToDevice);

    dim3 threads(WARP_SIZE, 1); // 1 warp per thread block
    dim3 blocks(M / TILE_DIM, N / TILE_DIM);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    WMMAKernel<<<blocks, threads>>>(d_A, d_B, d_C, d_D, M, N, K);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    printf("GPU execution time: %.3f ms\n", ms);
    printf("TFLOPS: %.2f\n", ((double)M * N * K * 2) / (ms * 1e6));

    // Copy result back to host and print
    std::vector<float> h_D(size_C);
    cudaMemcpy(h_D.data(), d_D, size_C * sizeof(float), cudaMemcpyDeviceToHost);

    printf("Result matrix D = A*B + C (%dx%d):\n", M, N);
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            printf("%.3f ", h_D[i * N + j]);
        }
        printf("\n");
    }

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaFree(d_D);
    return 0;
}
