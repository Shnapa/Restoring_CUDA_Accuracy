#ifndef MATRIX_PARSER_HPP
#define MATRIX_PARSER_HPP

#include <array>
#include <iostream>
#include <vector>
#include <random>
#include <fstream>
#include <sstream>

inline std::vector<std::tuple<size_t, size_t, size_t>> matrix_sizes = {
    {1000, 500, 800},
    {2000, 1000, 1500},
    {5000, 2500, 3000},
    {10000, 5000, 8000},
    {15000, 7500, 10000},
    {20000, 12500, 15000},
    {25000, 15000, 17500},
    {35000, 17500, 20000},
    {40000, 20000, 22500},
};

inline std::vector<std::string> filePaths{
    "../data/matrix_1000_500_800.txt",
    "../data/matrix_2000_1000_1500.txt",
    "../data/matrix_5000_2500_3000.txt",
    "../data/matrix_10000_5000_8000.txt",
    "../data/matrix_15000_7500_10000.txt",
    "../data/matrix_20000_12500_15000.txt",
    "../data/matrix_25000_15000_17500.txt",
    "../data/matrix_35000_17500_20000.txt",
    "../data/matrix_40000_20000_22500.txt",
};

int loadMatrixFromFile(const std::string& filename, std::vector<std::vector<float>>& matrix);

int loadMatricesFromFileArray(  const std::string &filePath,
                                float* A,
                                size_t A_elements,
                                float* B,
                                size_t B_elements);


void parseDimensions(   const std::string& filePath,
                        size_t &m,
                        size_t &n,
                        size_t &k);

#endif
