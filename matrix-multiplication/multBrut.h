//
// Created by Admin on 02.04.2025.
//

#ifndef MULTBRUT_H
#define MULTBRUT_H

#include <vector>
#include <iostream>

using Matrix = std::vector<int>;
Matrix multiplication(const Matrix& A, size_t m1, size_t n1, const Matrix& B, size_t m2, size_t n2);
void printMatrix(const Matrix& mat, size_t rows, size_t cols);


#endif //MULTBRUT_H
