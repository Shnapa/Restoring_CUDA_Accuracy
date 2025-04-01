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
    {10000, 10000, 10000},
    {50000, 10000, 40000},
};

inline std::vector<std::string> filePaths{
    "../data/matrix_1000_500_800.txt",
    "../data/matrix_2000_1000_1500.txt",
    "../data/matrix_5000_2500_3000.txt",
    "../data/matrix_10000_5000_8000.txt",
    "../data/matrix_10000_10000_10000.txt",
    "../data/matrix_50000_10000_40000.txt"
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
