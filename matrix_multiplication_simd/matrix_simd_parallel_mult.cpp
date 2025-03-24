#include <iostream>
#include <vector>
#include <random>
#include <immintrin.h>
#include <iomanip>
#include <omp.h>
#include <chrono>
#include <fstream>

#define BLOCK_SIZE 64

void generateRandomMatrix(float* matrix, int size) {
    int totalElements = size * size;

    #pragma omp parallel
    {
        std::mt19937 gen(42 + omp_get_thread_num());
        std::uniform_real_distribution<float> dist(1.0f, 10.0f);

        #pragma omp for
        for (int i = 0; i < totalElements; i++) {
            matrix[i] = dist(gen);
        }
    }
}

void simdMatrixMultiply(const float* A, const float* B, float* C, int size) {
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < size; i += BLOCK_SIZE) {
        for (int j = 0; j < size; j += BLOCK_SIZE) {
            for (int k = 0; k < size; k += BLOCK_SIZE) {

                for (int ii = i; ii < i + BLOCK_SIZE && ii < size; ii++) {
                    for (int jj = j; jj < j + BLOCK_SIZE && jj < size; jj += 8) {
                        __m256 sum = _mm256_setzero_ps();

                        for (int kk = k; kk < k + BLOCK_SIZE && kk < size; kk++) {
                            __m256 a = _mm256_set1_ps(A[ii * size + kk]);
                            __m256 b = _mm256_loadu_ps(&B[kk * size + jj]);
                            sum = _mm256_fmadd_ps(a, b, sum);
                        }

                        _mm256_storeu_ps(&C[ii * size + jj], sum);
                    }

                    for (int jj = (j + BLOCK_SIZE) / 8 * 8; jj < j + BLOCK_SIZE && jj < size; jj++) {
                        float sum = 0.0f;
                        for (int kk = k; kk < k + BLOCK_SIZE && kk < size; kk++) {
                            sum += A[ii * size + kk] * B[kk * size + jj];
                        }
                        C[ii * size + jj] = sum;
                    }
                }
            }
        }
    }
}

void printMatrix(const float* matrix, int size, int maxPrint = 10) {
    for (int i = 0; i < std::min(size, maxPrint); i++) {
        for (int j = 0; j < std::min(size, maxPrint); j++) {
            std::cout << std::setw(6) << std::fixed << std::setprecision(2) << matrix[i * size + j] << " ";
        }
        if (size > maxPrint) std::cout << "...";
        std::cout << std::endl;
    }
    if (size > maxPrint) std::cout << "... (hidden remaining rows) ..." << std::endl;
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

    for (int i = 0; i < 4; i++) {
        int N = sizes[i];
        size = N * N * sizeof(float);

        float* A = new float[size];
        float* B = new float[size];
        float* C = new float[size]();

        generateRandomMatrix(A, N);
        generateRandomMatrix(B, N);

        auto start = std::chrono::high_resolution_clock::now();

        simdMatrixMultiply(A, B, C, N);

        auto end = std::chrono::high_resolution_clock::now();

        std::cout << "\nMatrix C (First 10x10 elements for size " << N << "):\n";
        printMatrix(C, N);

        std::chrono::duration<double> duration = end - start;
        writeExecutionTimeToFile(duration.count(), N, "execution_time_simd_parallel.txt");

        delete[] A;
        delete[] B;
        delete[] C;
    }

    return 0;
}
