#include <iostream>
#include <vector>
#include <random>
#include <immintrin.h>
#include <iomanip>
#include <chrono>
#include "matrixParser.h"
#include "timeMeasurement.h"

#define BLOCK_SIZE 64

void simdMatrixMultiply(const float* A, const float* B, float* C, int m, int n, int k) {
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < m; i += BLOCK_SIZE) {
        for (int j = 0; j < k; j += BLOCK_SIZE) {
            for (int kk = 0; kk < n; kk += BLOCK_SIZE) {

                for (int ii = i; ii < i + BLOCK_SIZE && ii < m; ii++) {
                    for (int jj = j; jj < j + BLOCK_SIZE && jj < k; jj += 8) {
                        __m256 sum = _mm256_setzero_ps();

                        for (int kk_iter = kk; kk_iter < kk + BLOCK_SIZE && kk_iter < n; kk_iter++) {
                            __m256 a = _mm256_set1_ps(A[ii * n + kk_iter]);
                            __m256 b = _mm256_loadu_ps(&B[kk_iter * k + jj]);
                            sum = _mm256_fmadd_ps(a, b, sum);
                        }

                        _mm256_storeu_ps(&C[ii * k + jj], sum);
                    }

                    for (int jj = (j + BLOCK_SIZE) / 8 * 8; jj < j + BLOCK_SIZE && jj < k; jj++) {
                        float sum = 0.0f;
                        for (int kk_iter = kk; kk_iter < kk + BLOCK_SIZE && kk_iter < n; kk_iter++) {
                            sum += A[ii * n + kk_iter] * B[kk_iter * k + jj];
                        }
                        C[ii * k + jj] = sum;
                    }
                }
            }
        }
    }
}

int main(int argc, char* argv[])
{
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <A_matrix_path> <B_matrix_path>" << std::endl;
        return 1;
    }

    std::string A_matrix_path = argv[1];
    std::string B_matrix_path = argv[2];

    std::vector<std::vector<float>> A = loadMatrixFromFile(A_matrix_path);
    std::vector<std::vector<float>> B = loadMatrixFromFile(B_matrix_path);

    int m = A.size();
    int n = A[0].size();
    int k = B[0].size();

    float* A_flat = new float[m * n];
    float* B_flat = new float[n * k];
    float* C_flat = new float[m * k]();

    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            A_flat[i * n + j] = A[i][j];
        }
    }

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < k; ++j) {
            B_flat[i * k + j] = B[i][j];
        }
    }

    auto start = get_current_time_fenced();
    simdMatrixMultiply(A_flat, B_flat, C_flat, m, n, k);
    auto end = get_current_time_fenced();

    std::cout << "Execution time: " << to_ms(end-start) << " ms" << std::endl;

    delete[] A_flat;
    delete[] B_flat;
    delete[] C_flat;
    return 0;
}

