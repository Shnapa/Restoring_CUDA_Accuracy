//
// Created by Admin on 02.04.2025.
//

#include "testing.h"
#include <vector>
#include <string>
#include "src/matrixParser.cpp"
#include "src/generateMatrices.cpp"

void generate_matrices(size_t m, size_t n, float* A) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(-1000.0, 1000.0);

    size_t totalA = m * n;
    for (size_t k = 0; k < totalA; ++k) {
        A[k] = dist(gen);
    }
}

bool compare_matrices(float* A, float* B, size_t m, size_t n, float epsilon) {
    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < n; ++j) {
            if (std::fabs(A[i * n + j] - B[i * n + j]) > epsilon) {
                return false; // Якщо різниця більша за epsilon
            }
        }
    }
    return true;
}



void write_matrices_to_file(const float* matrix1, const float* matrix2, size_t m, size_t n, size_t p, const std::string& filename) {
    std::ofstream outfile(filename);
    if (!outfile.is_open()) {
        std::cerr << "Error opening file: " << filename << std::endl;
        return;
    }

    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < n; ++j) {
            outfile << matrix1[i * n + j];
            if (j < n - 1) {
                outfile << " ";
            }
        }
        if (i < m - 1) {
            outfile << " ";
        }
    }

    outfile << "\n";

    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < p; ++j) {
            outfile << matrix2[i * p + j];
            if (j < p - 1) {
                outfile << " ";
            }
        }
        if (i < n - 1) {
            outfile << " ";
        }
    }

    outfile.close();
}


