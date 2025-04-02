//
// Created by Admin on 02.04.2025.
//

#ifndef MULTBRUT_H
#define MULTBRUT_H

#include <vector>
#include <iostream>

using Matrix = std::vector<int>;
Matrix multiplication(Matrix a, Matrix b);
void printMatrix(const Matrix& mat, size_t rows, size_t cols);


#endif //MULTBRUT_H
