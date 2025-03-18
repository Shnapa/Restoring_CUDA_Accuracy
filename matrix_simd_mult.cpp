#include <iostream>
#include <vector>
#include <random>
#include <immintrin.h>
#include <iomanip>
#include <omp.h>

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
    if (size > maxPrint) std::cout << "... (hidden remain rows) ..." << std::endl;
}

int main() {
    const int size = 10000;

    float* A = new float[size * size];
    float* B = new float[size * size];
    float* C = new float[size * size]();

    generateRandomMatrix(A, size);
    generateRandomMatrix(B, size);

    simdMatrixMultiply(A, B, C, size);

    std::cout << "\nMatrix C\n";
    printMatrix(C, size);

    delete[] A;
    delete[] B;
    delete[] C;

    return 0;
}
