//
// Created by gllek-pc on 3/30/25.
//

#include <regex>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <cstdlib>
#include "../include/matrixParser.h"

void loadMatrices_RR(const std::string &filePath,
                     std::vector<float> &A,
                     std::vector<float> &B)
{
    size_t m, k, n;
    parseDimensions(filePath, m, k, n);
    std::ifstream fin(filePath);
    if (!fin.is_open()) std::exit(EXIT_FAILURE);
    std::string line;
    std::getline(fin, line);
    std::istringstream isa(line);
    float v;
    size_t i = 0;
    while (isa >> v && i < A.size()) A[i++] = v;
    std::getline(fin, line);
    isa.clear();
    isa.str(line);
    i = 0;
    while (isa >> v && i < B.size()) B[i++] = v;
}

void loadMatrices_RC(const std::string &filePath,
                    std::vector<float> &A,
                    std::vector<float> &B)
{
    size_t m,k,n;
    parseDimensions(filePath, m, k, n);
    std::ifstream fin(filePath);
    if (!fin.is_open()) std::exit(EXIT_FAILURE);
    std::string line;

    std::getline(fin, line);
    std::istringstream isa(line);
    float v; size_t i=0;
    while (isa >> v && i < A.size()) A[i++] = v;

    std::getline(fin, line);
    isa.clear(); isa.str(line);
    std::vector<float> tmp(B.size());
    i=0;
    while (isa >> v && i < tmp.size()) tmp[i++] = v;
    for (size_t col=0; col<n; ++col)
      for (size_t row=0; row<k; ++row)
        B[col*k + row] = tmp[row*n + col];
}

void loadMatrices_CC(const std::string &filePath,
                    std::vector<float> &A,
                    std::vector<float> &B)
{
    size_t m,k,n;
    parseDimensions(filePath, m, k, n);
    std::ifstream fin(filePath);
    if (!fin.is_open()) std::exit(EXIT_FAILURE);
    std::string line;

    std::getline(fin, line);
    std::istringstream isa(line);
    std::vector<float> tmpA(A.size());
    float v; size_t i=0;
    while (isa >> v && i < tmpA.size()) tmpA[i++] = v;
    for (size_t col=0; col<k; ++col)
      for (size_t row=0; row<m; ++row)
        A[col*m + row] = tmpA[row*k + col];

    std::getline(fin, line);
    isa.clear(); isa.str(line);
    std::vector<float> tmpB(B.size());
    i=0;
    while (isa >> v && i < tmpB.size()) tmpB[i++] = v;
    for (size_t col=0; col<n; ++col)
      for (size_t row=0; row<k; ++row)
        B[col*k + row] = tmpB[row*n + col];
}

void parseDimensions(const std::string& filePath, size_t &m, size_t &k, size_t &n) {
    const std::regex pattern(R"(.*_(\d+)_(\d+)_(\d+)\.txt)");
    if (std::smatch match; std::regex_match(filePath, match, pattern)) {
        m = std::stoi(match[1]);
        k = std::stoi(match[2]);
        n = std::stoi(match[3]);
    } else {
        throw std::runtime_error("Filename does not match expected format");
    }
}