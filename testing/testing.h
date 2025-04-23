//
// Created by Admin on 02.04.2025.
//

#ifndef TESTING_H
#define TESTING_H

#include <vector>
#include <string>
#include "include/matrixParser.cpp"
#include "src/generateMatrices.cpp"

void generate_matrices(size_t m, size_t n, float* A);
bool compare_matrices(float* A, float* B, size_t m, size_t n, float epsilon = 1e-8);
void write_matrices_to_file(const float* matrix1, const float* matrix2, size_t m, size_t n, const std::string& filename);


#endif //TESTING_H
