#ifndef MATRIX_PARSER_HPP
#define MATRIX_PARSER_HPP

#include <array>
#include <iostream>
#include <vector>
#include <random>
#include <fstream>
#include <sstream>
#include <tuple>
inline std::vector<std::tuple<size_t, size_t, size_t>> matrix_sizes = {
    // {5, 2, 4}
    {250, 100, 80},
    {500, 250, 150},
    {1000, 500, 800},
    {2000, 1000, 1500},
    {5000, 2500, 3000},
    // {10000, 5000, 8000},
    // {15000, 7500, 10000},
    // {20000, 12500, 15000},
    // {25000, 15000, 17500},
    // {35000, 17500, 20000},
    // {41000, 13000, 24000},
    // {42000, 13000, 24000},
    // {42000, 14000, 24000},
};

inline std::vector<std::string> filePaths{
    "../data/matrix_250_100_80.txt",
    "../data/matrix_500_250_150.txt",
    "../data/matrix_1000_500_800.txt",
    "../data/matrix_2000_1000_1500.txt",
    "../data/matrix_5000_2500_3000.txt",
    // "../data/matrix_10000_5000_8000.txt",
    // "../data/matrix_15000_7500_10000.txt",
    // "../data/matrix_20000_12500_15000.txt",
    // "../data/matrix_25000_15000_17500.txt",
    // "../data/matrix_35000_17500_20000.txt",
    // "../data/matrix_42000_14000_24000.txt",
};

void loadMatrices_CC(const std::string &filePath,
                    std::vector<float> &A,
                    std::vector<float> &B);

void loadMatrices_RR(const std::string &filePath,
                    std::vector<float> &A,
                    std::vector<float> &B);

void loadMatrices_RC(const std::string &filePath,
                    std::vector<float> &A,
                    std::vector<float> &B);

void parseDimensions(   const std::string& filePath,
                        size_t &m,
                        size_t &k,
                        size_t &n );

#endif
