#include <iostream>
#include <vector>
#include <random>
#include <immintrin.h>
#include <iomanip>

std::vector<std::vector<float>> padMatrix(const std::vector<std::vector<float>>& matrix, int simdWidth = 8) {
    int originalRows = matrix.size();
    int originalCols = matrix[0].size();
    int paddedRows = (originalRows + simdWidth - 1) / simdWidth * simdWidth;
    int paddedCols = (originalCols + simdWidth - 1) / simdWidth * simdWidth;
    std::vector<std::vector<float>> paddedMatrix(paddedRows, std::vector<float>(paddedCols, 0.0f));
    for (int i = 0; i < originalRows; i++) {
        for (int j = 0; j < originalCols; j++) {
            paddedMatrix[i][j] = matrix[i][j];
        }
    }
    return paddedMatrix;
}

std::vector<std::vector<float>> removePadding(const std::vector<std::vector<float>>& paddedMatrix, int originalRows, int originalCols) {
    std::vector<std::vector<float>> originalMatrix(originalRows, std::vector<float>(originalCols, 0.0f));
    for (int i = 0; i < originalRows; i++) {
        for (int j = 0; j < originalCols; j++) {
            originalMatrix[i][j] = paddedMatrix[i][j];
        }
    }
    return originalMatrix;
}

std::vector<std::vector<float>> simdMatrixMultiply(const std::vector<std::vector<float>>& A, const std::vector<std::vector<float>>& B) {
    std::vector<std::vector<float>> paddedA = padMatrix(A);
    std::vector<std::vector<float>> paddedB = padMatrix(B);
    int N = paddedA.size();
    int M = paddedB.size();
    int P = paddedB[0].size();
    std::vector<std::vector<float>> C(N, std::vector<float>(P, 0.0f));

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < P; j += 8) {
            __m256 sum = _mm256_setzero_ps();
            for (int k = 0; k < M; k++) {
                __m256 a = _mm256_set1_ps(paddedA[i][k]);
                __m256 b = _mm256_loadu_ps(paddedB[k].data() + j);
                sum = _mm256_fmadd_ps(a, b, sum);
            }
            _mm256_storeu_ps(C[i].data() + j, sum);
        }
    }

    return removePadding(C, A.size(), B[0].size());
}

std::vector<std::vector<float>> generateRandomMatrix(int size) {
    std::vector<std::vector<float>> matrix(size, std::vector<float>(size));
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(1.0f, 10.0f); // Random values between 1 and 10

    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            matrix[i][j] = dist(gen);
        }
    }
    return matrix;
}

void printMatrix(const std::vector<std::vector<float>>& matrix, int maxPrint = 10) {
    int rows = matrix.size();
    int cols = matrix[0].size();

    for (int i = 0; i < std::min(rows, maxPrint); i++) {
        for (int j = 0; j < std::min(cols, maxPrint); j++) {
            std::cout << std::setw(6) << std::fixed << std::setprecision(2) << matrix[i][j] << " ";
        }
        if (cols > maxPrint) std::cout << "...";
        std::cout << std::endl;
    }
    if (rows > maxPrint) std::cout << "... (remaining rows hidden) ..." << std::endl;
}

int main() {
    const int size = 1000;

    std::vector<std::vector<float>> A = generateRandomMatrix(size);
    std::vector<std::vector<float>> B = generateRandomMatrix(size);

    std::cout << "Matrix A (First 10x10 elements):\n";
    printMatrix(A);

    std::cout << "\nMatrix B (First 10x10 elements):\n";
    printMatrix(B);

    std::vector<std::vector<float>> C = simdMatrixMultiply(A, B);

    std::cout << "\nResult Matrix C (First 10x10 elements):\n";
    printMatrix(C);

    return 0;
}
