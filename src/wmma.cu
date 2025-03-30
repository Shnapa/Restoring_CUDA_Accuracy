#include <cuda_fp16.h>
#include <mma.h>
#include <cstdio>

using namespace nvcuda;

#define M 16
#define N 16
#define K 16

__global__ void matmul_wmma(half *A, half *B, float *D) {
    wmma::fragment<wmma::matrix_a, M, N, K, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, M, N, K, half, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, M, N, K, float> acc_frag;

    wmma::fill_fragment(acc_frag, 0.0f);

    wmma::load_matrix_sync(a_frag, A, K);
    wmma::load_matrix_sync(b_frag, B, N);

    wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
    wmma::store_matrix_sync(D, acc_frag, N, wmma::mem_row_major);
}

int main() {
    half *A, *B;
    float *D;

    size_t sizeA = M * K * sizeof(half);
    size_t sizeB = K * N * sizeof(half);
    size_t sizeD = M * N * sizeof(float);

    cudaMallocManaged(&A, sizeA);
    cudaMallocManaged(&B, sizeB);
    cudaMallocManaged(&D, sizeD);

    for (int i = 0; i < M * K; i++)
        A[i] = __float2half(1.0f); // всі елементи = 1.0

    for (int i = 0; i < K * N; i++)
        B[i] = __float2half(1.0f); // всі елементи = 1.0

    matmul_wmma<<<1, 32>>>(A, B, D);
    cudaDeviceSynchronize();

    printf("Result matrix D:\n");
    for (int i = 0; i < 16; i++) {
        for (int j = 0; j < 16; j++) {
            printf("%.1f ", D[i * 16 + j]);
        }
        printf("\n");
    }

    cudaFree(A);
    cudaFree(B);
    cudaFree(D);

    return 0;
}
