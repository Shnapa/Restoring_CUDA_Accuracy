#include <iostream>
#include <vector>
#include <cuda_runtime.h>

__global__ void matrixMultiplicationKernel(double *A, double *B, double *C, int n, int k, int m) {
    int ROW = blockIdx.x * blockDim.x + threadIdx.x;
    int COL = blockIdx.y * blockDim.x + threadIdx.x;

    double currentSum = 0.0;

    if (ROW < m && COL < n) {
        for (size_t i = 0; i < k; i++) {
            currentSum += A[ROW * k + i] * B[COL * m + i];
        }
    }

    C[COL * m + n] = currentSum;
}

void matrixMultiplication(double *A, double *B, double *C, int n, int k, int m) {
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((m + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (n + threadsPerBlock.y - 1) / threadsPerBlock.y);

    matrixMultiplicationKernel<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, n, k, m);
}

int main() {
    int n = 10;
    int k = 5;
    int m = 8;

    std::vector<double> h_A(n * k), h_B(k * m), h_C(n * m, 0.0);

    for (int i = 0; i < n * k; i++) h_A[i] = rand() % 10;
    for (int i = 0; i < k * m; i++) h_B[i] = rand() % 10;

    double *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, sizeof(double) * n * k);
    cudaMalloc(&d_B, sizeof(double) * n * m);
    cudaMalloc(&d_C, sizeof(double) * n * m);

    cudaMemcpy(d_A, h_A.data(), sizeof(double) * n * k, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B.data(), sizeof(double) * k * m, cudaMemcpyHostToDevice);

    matrixMultiplication(d_A, d_B, d_C, n, k, m);

    cudaMemcpy(h_C.data(), d_C, n * m * sizeof(double), cudaMemcpyDeviceToHost);

    std::cout << "Result matrix C (first 5 elements):\n";
    for (int i = 0; i < std::min(5, n * m); i++) {
        std::cout << h_C[i] << " ";
    }
    std::cout << std::endl;

    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;

}