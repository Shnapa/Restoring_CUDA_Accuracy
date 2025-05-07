#include <cstdio>

#include "compare.h"
#include "matrixParser.h"
#include <string>
#include <cuda_fp16.h>
#include <mma.h>

#define SCALE 2048.0f
#define TILE_SIZE 16
#define WARP_SIZE 32
using namespace nvcuda::wmma;
__global__ void restoreAccuracy(const __half* __restrict__ A,
                                const __half* __restrict__ B,
                                float* __restrict__ C,
                                const size_t padded_M,
                                const size_t padded_N,
                                const size_t padded_K)
{
    const size_t warpM = blockIdx.x;
    const size_t warpN = blockIdx.y;
    const size_t lane  = threadIdx.x;

    if (warpM * TILE_SIZE >= padded_M || warpN * TILE_SIZE >= padded_N)
        return;

    fragment<matrix_a, TILE_SIZE, TILE_SIZE, TILE_SIZE, __half, row_major> a_frag, da_frag;
    fragment<matrix_b, TILE_SIZE, TILE_SIZE, TILE_SIZE, __half, row_major> b_frag, db_frag;
    fragment<accumulator, TILE_SIZE, TILE_SIZE, TILE_SIZE, float> tmp_frag, dc_frag, c_frag;

    fill_fragment(c_frag,  0.0f);
    fill_fragment(dc_frag, 0.0f);

    __shared__ __half shA[TILE_SIZE * TILE_SIZE],
                         shB[TILE_SIZE * TILE_SIZE],
                         shDA[TILE_SIZE * TILE_SIZE],
                         shDB[TILE_SIZE * TILE_SIZE];

    for (size_t tileK = 0; tileK < padded_K; tileK += TILE_SIZE) {
        fill_fragment(tmp_frag, 0.0f);

        for (size_t idx = lane; idx < TILE_SIZE * TILE_SIZE; idx += WARP_SIZE) {
            const size_t r = idx / TILE_SIZE;
            const size_t c = idx % TILE_SIZE;
            const size_t aIdx = (warpM * TILE_SIZE + r) * padded_K + (tileK + c);
            const size_t bIdx = (tileK + r) * padded_N + (warpN * TILE_SIZE + c);
            const float aVal = __half2float(A[aIdx]);
            const float bVal = __half2float(B[bIdx]);
            const __half  a16  = __float2half(aVal);
            const __half  b16  = __float2half(bVal);
            const __half  da   = __float2half((aVal - __half2float(a16)) * SCALE);
            const __half  db   = __float2half((bVal - __half2float(b16)) * SCALE);
            shA[idx]  = a16;
            shB[idx]  = b16;
            shDA[idx] = da;
            shDB[idx] = db;
        }
        __syncthreads();

        load_matrix_sync(a_frag,  shA,  TILE_SIZE);
        load_matrix_sync(b_frag,  shB,  TILE_SIZE);
        load_matrix_sync(da_frag, shDA, TILE_SIZE);
        load_matrix_sync(db_frag, shDB, TILE_SIZE);

        mma_sync(tmp_frag, a_frag, b_frag, tmp_frag);
        mma_sync(dc_frag,   da_frag, b_frag,  dc_frag);
        mma_sync(dc_frag,   a_frag,  db_frag, dc_frag);

        for (int e = 0; e < c_frag.num_elements; ++e) {
            c_frag.x[e] += tmp_frag.x[e];
        }
    }

    for (int e = 0; e < c_frag.num_elements; ++e) {
        c_frag.x[e] += dc_frag.x[e] / SCALE;
    }

    const size_t row0 = warpM * TILE_SIZE;
    const size_t col0 = warpN * TILE_SIZE;
    float* tileC = C + row0 * padded_N + col0;
    store_matrix_sync(tileC, c_frag, padded_N, mem_row_major);
}

void wmmaMatrixMultiply(
    const float* h_A, const float* h_B, float* h_C,
    const size_t m, const size_t k, const size_t n,
    float& executionTime)
{
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

    for (size_t i = 0; i < padded_M * padded_N; i++) h_C_pad[i] = 0.0f;

    __half *d_A, *d_B;
    float  *d_C;
    cudaMalloc(&d_A, padded_M * padded_K * sizeof(__half));
    cudaMalloc(&d_B, padded_K * padded_N * sizeof(__half));
    cudaMalloc(&d_C, padded_M * padded_N * sizeof(float));

    cudaMemcpy(d_A, h_A_pad, padded_M * padded_K * sizeof(__half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B_pad, padded_K * padded_N * sizeof(__half), cudaMemcpyHostToDevice);
    cudaMemset(d_C, 0, padded_M * padded_N * sizeof(float));

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    dim3 threads(WARP_SIZE, 1);
    dim3 blocks(padded_M / TILE_SIZE, padded_N / TILE_SIZE);

    cudaEventRecord(start);
    restoreAccuracy<<<blocks, threads>>>(d_A, d_B, d_C, padded_M, padded_N, padded_K);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&executionTime, start, stop);

    cudaMemcpy(h_C_pad, d_C, padded_M * padded_N * sizeof(float), cudaMemcpyDeviceToHost);
    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < n; ++j) {
            h_C[i * n + j] = h_C_pad[i * padded_N + j];
        }
    }

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    delete[] h_A_pad;
    delete[] h_B_pad;
    delete[] h_C_pad;
}

int main(const int argc, char* argv[])
{
    if (argc != 2) {
        fprintf(stderr, "Usage: %s matrix_MxKxN.txt\n", argv[0]);
        return EXIT_FAILURE;
    }
    const std::string filename = argv[1];

    size_t m, k, n;
    parseDimensions(filename, m, k, n);

    std::vector<float> A_f(m*k), B_f(n*k), C_f(m*n, 0.0f);
    loadMatrices_RR(filename, A_f, B_f);

    float executionTime = 0.0f;
    wmmaMatrixMultiply(A_f.data(), B_f.data(), C_f.data(), m, k, n, executionTime);
    std::cout << "Execution time: " << executionTime << "ms" << std::endl;
    compare(C_f, m, k, n, filename);
}