#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include "timeMeasurement.h"
#include "matrixParser.h"

__global__ void matrixMultiplicationKernel(double *A, double *B, double *C, int n, int k, int m) {
    int ROW = blockIdx.y * blockDim.y + threadIdx.y;
    int COL = blockIdx.x * blockDim.x + threadIdx.x;

    double currentSum = 0.0;

    if (ROW < m && COL < n) {
        for (size_t i = 0; i < k; i++) {
            currentSum += A[ROW * k + i] * B[i * m + COL];
        }
    }

    C[ROW * m + COL] = currentSum;
}

void matrixMultiplication(double *A, double *B, double *C, int n, int k, int m) {
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((m + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (n + threadsPerBlock.y - 1) / threadsPerBlock.y);

    matrixMultiplicationKernel<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, n, k, m);
}

int main(int argc, char* argv[]){
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <matrix_A_file> <matrix_B_file>" << std::endl;
        return 1;
    }
    std::string A_file = argv[1];
    std::string B_file = argv[2];

    MatrixData matA = parseMatrix(A_file);
    MatrixData matB = parseMatrix(B_file);

    if (matA.dim2 != matB.dim1) {
        std::cerr << "Dimension mismatch: A columns (" << matA.dim2
                  << ") != B rows (" << matB.dim1 << ")" << std::endl;
        return 1;
    }

    int n = static_cast<int>(matA.dim1);
    int k = static_cast<int>(matA.dim2);
    int m = static_cast<int>(matB.dim2);

    std::vector<double>& h_A = matA.data;
    std::vector<double>& h_B = matB.data;
    std::vector<double> h_C(n * m, 0.0);

    double *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, sizeof(double) * n * k);
    cudaMalloc(&d_B, sizeof(double) * k * m);
    cudaMalloc(&d_C, sizeof(double) * n * m);

    cudaMemcpy(d_A, h_A.data(), sizeof(double) * n * k, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B.data(), sizeof(double) * k * m, cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    matrixMultiplication(d_A, d_B, d_C, n, k, m);

    cudaDeviceSynchronize();
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    float elapsed_ms = 0.0f;
    cudaEventElapsedTime(&elapsed_ms, start, stop);
    std::cout << "Elapsed time: " << std::fixed << elapsed_ms << " milisec" << std::endl;
    cudaMemcpy(h_C.data(), d_C, n * m * sizeof(double), cudaMemcpyDeviceToHost);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;

}

