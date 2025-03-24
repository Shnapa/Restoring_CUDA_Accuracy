#include <cuda_runtime.h>
#include <iostream>
#include <fstream>
#include <chrono>

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

void writeExecutionTimeToFile(double executionTime, int N, const std::string& filename) {
    std::ofstream file(filename, std::ios::app);

    if (file.is_open()) {
        if (file.tellp() == 0) {
            file << "Size, Execution Time (seconds)\n";
        }
        file << N << ", " << executionTime << "\n";
        file.close();
    } else {
        std::cerr << "Error opening file for writing execution time!" << std::endl;
    }
}

int main() {
    int sizes[] = {10, 100, 1000, 10000};
    size_t size;

    for (int i = 0; i < 3; i++) {
        int N = sizes[i];
        size = N * N * sizeof(float);

        float *h_A = (float*)malloc(size);
        float *h_B = (float*)malloc(size);
        float *h_C = (float*)malloc(size);

        for (int j = 0; j < N * N; j++) {
            h_A[j] = 1.0f;
            h_B[j] = 1.0f;
        }

        auto start = std::chrono::high_resolution_clock::now();
        multiplyMatrices(h_A, h_B, h_C, N);
        auto end = std::chrono::high_resolution_clock::now();

        std::chrono::duration<double> duration = end - start;
        writeExecutionTimeToFile(duration.count(), N, "execution_time.txt");

        std::cout << "C[0][0] = " << h_C[0] << std::endl;

        free(h_A);
        free(h_B);
        free(h_C);
    }

    return 0;
}

