//
// Created by gllek-pc on 4/23/25.
//

#ifndef MATRIXPARSER2_H
#define MATRIXPARSER2_H
#include <array>
#include <iostream>
#include <vector>
#include <random>
#include <fstream>
#include <sstream>
#include <tuple>
inline std::vector<std::tuple<size_t, size_t, size_t>> matrix_sizes = {
    {  100,   100,   100},
    // {  1024,   1024,   1024},
    // {  2048,   2048,   2048},
    // {  4096,   4096,   4096},
    // {  8192,   8192,   8192},
    // { 16384,  16384,  16384},
    //
    // {  9999,    9999,    9999},
    // { 13331,   10007,   20011},
    // { 22222,   11111,    5555},
    //
    // { 32768,  16384,   8192},
    // { 40960,   8192,   4096},
    //
    // { 10000,  15000,  10000},
    // { 15000,  10000,  15000},
    // { 20000,  10000,   8000},
    // { 30000,   7000,  18000},
    // { 35000,  20000,   5000},
    //
    // { 24000,  24000,  24000},
    // { 25000,  25000,  25000},
    //
    // { 25000,  20000,  15000},
    // { 15000,  35000,  10000},
    // { 10000,  40000,   6000},
    //
    // { 42000,   2000,   2000},
    // {  2000,   2000,  42000},
    // {  2000,  42000,   2000},
    //
    // { 42000,  14000,  24000},
};

inline std::vector<std::string> filePaths = {
    "../data/matrix_1024_1024_1024.txt",
    "../data/matrix_2048_2048_2048.txt",
    "../data/matrix_4096_4096_4096.txt",
    "../data/matrix_8192_8192_8192.txt",
    "../data/matrix_16384_16384_16384.txt",

    "../data/matrix_9999_9999_9999.txt",
    "../data/matrix_13331_10007_20011.txt",
    "../data/matrix_22222_11111_5555.txt",

    "../data/matrix_32768_16384_8192.txt",
    "../data/matrix_40960_8192_4096.txt",

    "../data/matrix_10000_15000_10000.txt",
    "../data/matrix_15000_10000_15000.txt",
    "../data/matrix_20000_10000_8000.txt",
    "../data/matrix_30000_7000_18000.txt",
    "../data/matrix_35000_20000_5000.txt",

    "../data/matrix_24000_24000_24000.txt",
    "../data/matrix_25000_25000_25000.txt",

    "../data/matrix_25000_20000_15000.txt",
    "../data/matrix_15000_35000_10000.txt",
    "../data/matrix_10000_40000_6000.txt",

    "../data/matrix_42000_2000_2000.txt",
    "../data/matrix_2000_2000_42000.txt",
    "../data/matrix_2000_42000_2000.txt",

    "../data/matrix_42000_14000_24000.txt",
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

#endif //MATRIXPARSER2_H
