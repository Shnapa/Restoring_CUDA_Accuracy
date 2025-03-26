#include <cuda_runtime.h>
#include "timeMeasurement.h"
#include <iostream>
#include "matrixParser.h"

#define TILE_SIZE 32

__global__ void matrixMultiplyTiled(float *A, float *B, float *C, int N) {
    // Спільна пам’ять для тайлів
    __shared__ float tileA[TILE_SIZE][TILE_SIZE];
    __shared__ float tileB[TILE_SIZE][TILE_SIZE];

    // Індекси поточного потоку
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    float sum = 0.0f;

    // Кількість підматриць, які потрібно обробити
    for (int t = 0; t < (N + TILE_SIZE - 1) / TILE_SIZE; t++) {
        // Завантажуємо дані в спільну пам’ять (перевіряємо межі)
        if (row < N && (t * TILE_SIZE + threadIdx.x) < N)
            tileA[threadIdx.y][threadIdx.x] = A[row * N + t * TILE_SIZE + threadIdx.x];
        else
            tileA[threadIdx.y][threadIdx.x] = 0.0f;

        if (col < N && (t * TILE_SIZE + threadIdx.y) < N)
            tileB[threadIdx.y][threadIdx.x] = B[(t * TILE_SIZE + threadIdx.y) * N + col];
        else
            tileB[threadIdx.y][threadIdx.x] = 0.0f;

        __syncthreads(); // Синхронізація потоків

        // Виконуємо множення для даного тайлу
        for (int i = 0; i < TILE_SIZE; i++)
            sum += tileA[threadIdx.y][i] * tileB[i][threadIdx.x];

        __syncthreads(); // Очікуємо завершення перед наступною ітерацією
    }

    // Записуємо результат у глобальну пам’ять
    if (row < N && col < N)
        C[row * N + col] = sum;
}

void multiplyMatrices(float *h_A, float *h_B, float *h_C, int N) {
    float *d_A, *d_B, *d_C;
    size_t size = N * N * sizeof(float);

    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);

    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(TILE_SIZE, TILE_SIZE);
    dim3 blocksPerGrid((N + TILE_SIZE - 1) / TILE_SIZE, (N + TILE_SIZE - 1) / TILE_SIZE);

    matrixMultiplyTiled<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);
    cudaDeviceSynchronize();

    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
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

    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, sizeof(float) * n * k);
    cudaMalloc(&d_B, sizeof(float) * k * m);
    cudaMalloc(&d_C, sizeof(float) * n * m);

    cudaMemcpy(d_A, h_A.data(), sizeof(float) * n * k, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B.data(), sizeof(float) * k * m, cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    multiplyMatrices(d_A, d_B, d_C, n);

    cudaDeviceSynchronize();
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    float elapsed_ms = 0.0f;
    cudaEventElapsedTime(&elapsed_ms, start, stop);
    std::cout << "Elapsed time: " << std::fixed << elapsed_ms << " milisec" << std::endl;

    cudaMemcpy(h_C.data(), d_C, n * m * sizeof(float), cudaMemcpyDeviceToHost);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;

}