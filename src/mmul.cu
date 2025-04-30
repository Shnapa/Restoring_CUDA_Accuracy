#include "mmul.cuh"
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <mma.h>

#define TILE_SIZE 16
#define WARP_SIZE 32

__global__ void cudaMulKernel(const float* A, const float* B, float* C,
                            const size_t m, const size_t k, const size_t n) {
    const size_t row = blockIdx.y * blockDim.y + threadIdx.y;
    const size_t col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < m && col < n) {
        float sum = 0.0f;
        for (size_t i = 0; i < k; ++i) {
            sum += A[row * k + i] * B[i * n + col];
        }
        C[row * n + col] = sum;
    }
}

__global__ void cudaMulOptKernel(const float* A, const float* B, float* C,
                               const size_t m, const size_t k, const size_t n) {
    __shared__ float tileA[TILE_SIZE][TILE_SIZE];
    __shared__ float tileB[TILE_SIZE][TILE_SIZE];

    const size_t row = blockIdx.y * TILE_SIZE + threadIdx.y;
    const size_t col = blockIdx.x * TILE_SIZE + threadIdx.x;
    float sum = 0.0f;

    const size_t numTiles = (k + TILE_SIZE - 1) / TILE_SIZE;
    for (size_t t = 0; t < numTiles; ++t) {
        const size_t a_col = t * TILE_SIZE + threadIdx.x;
        const size_t b_row = t * TILE_SIZE + threadIdx.y;

        tileA[threadIdx.y][threadIdx.x] = (row < m && a_col < k)
            ? A[row * k + a_col]
            : 0.0f;
        tileB[threadIdx.y][threadIdx.x] = (b_row < k && col < n)
            ? B[b_row * n + col]
            : 0.0f;

        __syncthreads();
        for (int i = 0; i < TILE_SIZE; ++i) {
            sum += tileA[threadIdx.y][i] * tileB[i][threadIdx.x];
        }
        __syncthreads();
    }

    if (row < m && col < n) {
        C[row * n + col] = sum;
    }
}

__global__ void wmmaMulKernel(const half* A, const half* B, float* C,
                            const size_t padded_M, const size_t padded_N, const size_t padded_K) {
    const size_t warpM = blockIdx.x;
    const size_t warpN = blockIdx.y;

    if (warpM * TILE_SIZE >= padded_M || warpN * TILE_SIZE >= padded_N)
        return;

    using namespace nvcuda::wmma;

    fragment<matrix_a, TILE_SIZE, TILE_SIZE, TILE_SIZE, half, row_major> a_frag;
    fragment<matrix_b, TILE_SIZE, TILE_SIZE, TILE_SIZE, half, row_major> b_frag;
    fragment<accumulator, TILE_SIZE, TILE_SIZE, TILE_SIZE, float> acc_frag;

    fill_fragment(acc_frag, 0.0f);

    for (size_t tileK = 0; tileK < padded_K; tileK += TILE_SIZE) {
        const half* tileA = A + warpM * TILE_SIZE * padded_K + tileK;
        const half* tileB = B + tileK * padded_N + warpN * TILE_SIZE;

        load_matrix_sync(a_frag, tileA, padded_K);
        load_matrix_sync(b_frag, tileB, padded_N);
        mma_sync(acc_frag, a_frag, b_frag, acc_frag);
    }

    float* tileC = C + warpM * TILE_SIZE * padded_N + warpN * TILE_SIZE;
    store_matrix_sync(tileC, acc_frag, padded_N, mem_row_major);
}

template <typename T>
void padMatrix(const T* src, T* dst,
              const size_t rows, const size_t cols,
              const size_t padded_rows, const size_t padded_cols,
              T zero = T(0)) {
    for (size_t i = 0; i < padded_rows * padded_cols; ++i) {
        dst[i] = zero;
    }

    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            dst[i * padded_cols + j] = src[i * cols + j];
        }
    }
}

void cudaMatrixMultiply(const float* h_A, const float* h_B, float* h_C,
                      const size_t m, const size_t k, const size_t n,
                      float& executionTime) {
    const size_t size_A = m * k * sizeof(float);
    const size_t size_B = k * n * sizeof(float);
    const size_t size_C = m * n * sizeof(float);

    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size_A);
    cudaMalloc(&d_B, size_B);
    cudaMalloc(&d_C, size_C);

    cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice);
    cudaMemset(d_C, 0, size_C);

    dim3 threadsPerBlock(TILE_SIZE, TILE_SIZE);
    dim3 blocksPerGrid((n + TILE_SIZE - 1) / TILE_SIZE,
                     (m + TILE_SIZE - 1) / TILE_SIZE);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    cudaMulKernel<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, m, k, n);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    cudaEventElapsedTime(&executionTime, start, stop);

    cudaMemcpy(h_C, d_C, size_C, cudaMemcpyDeviceToHost);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

void cudaMatrixMultiplyOptimized(const float* h_A, const float* h_B, float* h_C,
                               const size_t m, const size_t k, const size_t n,
                               float& executionTime) {
    const size_t size_A = m * k * sizeof(float);
    const size_t size_B = k * n * sizeof(float);
    const size_t size_C = m * n * sizeof(float);

    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size_A);
    cudaMalloc(&d_B, size_B);
    cudaMalloc(&d_C, size_C);

    cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice);
    cudaMemset(d_C, 0, size_C);

    dim3 threadsPerBlock(TILE_SIZE, TILE_SIZE);
    dim3 blocksPerGrid((n + TILE_SIZE - 1) / TILE_SIZE,
                     (m + TILE_SIZE - 1) / TILE_SIZE);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    cudaMulOptKernel<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, m, k, n);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    cudaEventElapsedTime(&executionTime, start, stop);

    cudaMemcpy(h_C, d_C, size_C, cudaMemcpyDeviceToHost);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

void cublasMatrixMultiply(const float* h_A, const float* h_B, float* h_C,
                        const size_t m, const size_t k, const size_t n,
                        float& executionTime) {
    const size_t size_A = m * k * sizeof(float);
    const size_t size_B = k * n * sizeof(float);
    const size_t size_C = m * n * sizeof(float);

    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size_A);
    cudaMalloc(&d_B, size_B);
    cudaMalloc(&d_C, size_C);

    cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice);
    cudaMemset(d_C, 0, size_C);

    cublasHandle_t handle;
    cublasCreate(&handle);

    constexpr float alpha = 1.0f;
    constexpr float beta = 0.0f;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    cublasGemmEx(handle,
               CUBLAS_OP_N, CUBLAS_OP_N,
               n, m, k,
               &alpha,
               d_B, CUDA_R_32F, n,
               d_A, CUDA_R_32F, k,
               &beta,
               d_C, CUDA_R_32F, n,
               CUBLAS_COMPUTE_32F,
               CUBLAS_GEMM_DEFAULT);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    cudaEventElapsedTime(&executionTime, start, stop);

    cudaMemcpy(h_C, d_C, size_C, cudaMemcpyDeviceToHost);

    cublasDestroy(handle);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

void wmmaMatrixMultiply(const float* h_A, const float* h_B, float* h_C,
                       const size_t m, const size_t k, const size_t n,
                       float& executionTime) {
    const size_t padded_M = (m + TILE_SIZE - 1) / TILE_SIZE * TILE_SIZE;
    const size_t padded_K = (k + TILE_SIZE - 1) / TILE_SIZE * TILE_SIZE;
    const size_t padded_N = (n + TILE_SIZE - 1) / TILE_SIZE * TILE_SIZE;

    const auto h_A_pad = new __half[padded_M * padded_K];
    const auto h_B_pad = new half[padded_K * padded_N];
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

    half *d_A, *d_B;
    float *d_C;
    cudaMalloc(&d_A, padded_M * padded_K * sizeof(half));
    cudaMalloc(&d_B, padded_K * padded_N * sizeof(half));
    cudaMalloc(&d_C, padded_M * padded_N * sizeof(float));

    cudaMemcpy(d_A, h_A_pad, padded_M * padded_K * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B_pad, padded_K * padded_N * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemset(d_C, 0, padded_M * padded_N * sizeof(float));

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    dim3 threads(WARP_SIZE, 1);
    dim3 blocks(padded_M / TILE_SIZE, padded_N / TILE_SIZE);

    cudaEventRecord(start);

    wmmaMulKernel<<<blocks, threads>>>(d_A, d_B, d_C, padded_M, padded_N, padded_K);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    cudaEventElapsedTime(&executionTime, start, stop);

    cudaMemcpy(h_C_pad, d_C, padded_M * padded_N * sizeof(float), cudaMemcpyDeviceToHost);

    for (size_t i = 0; i < m; i++) {
        for (size_t j = 0; j < n; j++) {
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