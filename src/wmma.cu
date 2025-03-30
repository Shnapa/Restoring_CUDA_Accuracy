#include <cuda_fp16.h>
#include <mma.h>
#include <cstdio>

using namespace nvcuda;

#define M 16
#define N 16
#define K 16

#define M_TILES 4
#define N_TILES 4
#define K_TILES 4

#define M_TOTAL (M * M_TILES)
#define N_TOTAL (N * N_TILES)
#define K_TOTAL (K * K_TILES)

__global__ void matmul_wmma(half *A, half *B, float *C, float *D) {
    int warpId = threadIdx.x / 32;
    int blockRow = blockIdx.y;
    int blockCol = blockIdx.x;

    int row = (blockRow * blockDim.y + warpId) * M;
    int col = blockCol * N;

    if (row >= M_TOTAL || col >= N_TOTAL) return;

    wmma::fragment<wmma::matrix_a, M, N, K, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, M, N, K, half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, M, N, K, float> acc_frag;
    wmma::fragment<wmma::accumulator, M, N, K, float> c_frag;

    wmma::fill_fragment(acc_frag, 0.0f);

    for (int tileK = 0; tileK < K_TOTAL; tileK += K) {
        wmma::load_matrix_sync(a_frag, A + row * K_TOTAL + tileK, K_TOTAL);
        wmma::load_matrix_sync(b_frag, B + tileK * N_TOTAL + col, N_TOTAL);
        wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
    }

    wmma::load_matrix_sync(c_frag, C + row * N_TOTAL + col, N_TOTAL, wmma::mem_row_major);
    for (int i = 0; i < c_frag.num_elements; ++i) {
        acc_frag.x[i] += c_frag.x[i];
    }

    wmma::store_matrix_sync(D + row * N_TOTAL + col, acc_frag, N_TOTAL, wmma::mem_row_major);
}

int main() {
    half *A, *B;
    float *C, *D;

    cudaMallocManaged(&A, sizeof(half) * M_TOTAL * K_TOTAL);
    cudaMallocManaged(&B, sizeof(half) * K_TOTAL * N_TOTAL);
    cudaMallocManaged(&C, sizeof(float) * M_TOTAL * N_TOTAL);
    cudaMallocManaged(&D, sizeof(float) * M_TOTAL * N_TOTAL);

    for (int i = 0; i < M_TOTAL * K_TOTAL; i++) A[i] = __float2half(1.0f);
    for (int i = 0; i < K_TOTAL * N_TOTAL; i++) B[i] = __float2half(1.0f);
    for (int i = 0; i < M_TOTAL * N_TOTAL; i++) C[i] = 1.0f;

    dim3 blockDim(32 * 4);
    dim3 gridDim(N_TILES, M_TILES);

    matmul_wmma<<<gridDim, blockDim>>>(A, B, C, D);
    cudaDeviceSynchronize();

    printf("Result matrix D (перші 10x10 елементів):\n");
    for (int i = 0; i < 10; i++) {
        for (int j = 0; j < 10; j++) {
            printf("%.1f ", D[i * N_TOTAL + j]);
        }
        printf("\n");
    }

    cudaFree(A);
    cudaFree(B);
    cudaFree(C);
    cudaFree(D);

    return 0;
}
