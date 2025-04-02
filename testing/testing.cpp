//
// Created by Admin on 25.03.2025.
//

#include "testing.h"

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <ctime>
#include <cstdlib>

std::vector<std::vector<int>> generateMatrix(int rows, int cols, int minVal, int maxVal) {
    std::vector<std::vector<int>> matrix(rows, std::vector<int>(cols));

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            matrix[i][j] = minVal + rand() % (maxVal - minVal + 1);
        }
    }
    return matrix;
}
// Function to compare two matrices
bool compareMatrices(const std::vector<std::vector<int>>& A, const std::vector<std::vector<int>>& B) {
    if (A.size() != B.size() || A[0].size() != B[0].size()) {
        return false; // Different dimensions
    }

    for (size_t i = 0; i < A.size(); i++) {
        for (size_t j = 0; j < A[0].size(); j++) {
            if (A[i][j] != B[i][j]) {
                return false; // Found a difference
            }
        }
    }
    return true;
}

void printMatrix(const std::vector<std::vector<int>>& matrix) {
    for (const auto& row : matrix) {
        for (int val : row) {
            std::cout << val << " ";
        }
        std::cout << std::endl;
    }
}

std::vector<std::vector<int>> multiplyMatrices(const std::vector<std::vector<int>>& mat1, const std::vector<std::vector<int>>& mat2) {
    int rows1 = mat1.size();
    int cols1 = mat1[0].size();
    int rows2 = mat2.size();
    int cols2 = mat2[0].size();

    // Check if multiplication is possible (cols1 == rows2)
    if (cols1 != rows2) {
        std::cerr << "Matrix dimensions do not match for multiplication!" << std::endl;
        return {};
    }

    // Create the result matrix with appropriate dimensions
    std::vector<std::vector<int>> result(rows1, std::vector<int>(cols2, 0));

    // Perform matrix multiplication
    for (int i = 0; i < rows1; ++i) {
        for (int j = 0; j < cols2; ++j) {
            for (int k = 0; k < cols1; ++k) {
                result[i][j] += mat1[i][k] * mat2[k][j];
            }
        }
    }

    return result;
}

