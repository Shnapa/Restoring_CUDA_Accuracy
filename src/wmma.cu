#include <cuda_fp16.h>
#include <mma.h>
#include <cuda_runtime.h>
#include <vector>
#include <cstdlib>
#include "matrixParser.h"

using namespace nvcuda;

#define M 16
#define N 16
#define K 16

inline int loadHalfMatricesFromFileArray(const std::string &filePath, __half* A, size_t A_elements, __half* B, size_t B_elements) {
    std::ifstream file(filePath);
    std::string line;
    std::istringstream issA(line);
    size_t countA = 0;
    float value;
    while (issA >> value) {
        if (countA < A_elements) {
            A[countA++] = __float2half(value);
        } else {
            break;
        }
    }
    std::istringstream issB(line);
    size_t countB = 0;
    while (issB >> value) {
        if (countB < B_elements) {
            B[countB++] = __float2half(value);
        } else {
            break;
        }
    }
    return 0;
}

__global__ void matrixMultiplyWMMA(const half *A, const half *B, float *C, size_t m, size_t n, size_t k) {
    const size_t warpM = (blockIdx.y * blockDim.y + threadIdx.y) * M;
    const size_t warpN = (blockIdx.x * blockDim.x + threadIdx.x) * N;

    if (warpM >= m || warpN >= n) return;

    wmma::fragment<wmma::matrix_a, M, N, K, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, M, N, K, half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, M, N, K, float> acc_frag;

    wmma::fill_fragment(acc_frag, 0.0f);

    for (int i = 0; i < k; i += K) {
        wmma::load_matrix_sync(a_frag, A + warpM * k + i, k);
        wmma::load_matrix_sync(b_frag, B + i * n + warpN, n);
        wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
    }

    wmma::store_matrix_sync(C + warpM * n + warpN, acc_frag, n, wmma::mem_row_major);
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <matrix_file_path>" << std::endl;
        return 1;
    }
    const std::string filePath = argv[1];
    size_t m, n, k;
    parseDimensions(filePath, m, n, k);

    const size_t sizeA = m * n * sizeof(half);
    const size_t sizeB = n * k * sizeof(half);
    const size_t sizeC = m * k * sizeof(float);

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

    dim3 threadsPerBlock(2, 2);
    dim3 blocksPerGrid((n + N * threadsPerBlock.x - 1) / (N * threadsPerBlock.x),
                       (m + M * threadsPerBlock.y - 1) / (M * threadsPerBlock.y));

    matrixMultiplyWMMA<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, m, n, k);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    return 0;
}
