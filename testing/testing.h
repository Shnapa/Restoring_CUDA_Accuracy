//
// Created by Admin on 25.03.2025.
//

#ifndef TESTING_HPP
#define TESTING_HPP

#include <iostream>
#include <fstream>
#include <vector>
#include <sstream>
#include <cstdlib>

using Matrix = std::vector<int>;

bool compareMatrices(Matrix a, Matrix b, Matrix c);
Matrix multiplication(const Matrix& A, size_t m1, size_t n1, const Matrix& B, size_t m2, size_t n2);

#endif //TESTING_HPP
