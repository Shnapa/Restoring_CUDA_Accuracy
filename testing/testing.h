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

bool compareMatrices(const std::vector<std::vector<int>>& A, const std::vector<std::vector<int>>& B);
void printMatrix(const std::vector<std::vector<int>>& matrix);
std::vector<std::vector<int>> generateMatrix(int rows, int cols, int minVal = 0, int maxVal = 10);
std::vector<std::vector<int>> multiplyMatrices(const std::vector<std::vector<int>>& mat1, const std::vector<std::vector<int>>& mat2);
#endif //TESTING_HPP
