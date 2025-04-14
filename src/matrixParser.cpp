//
// Created by gllek-pc on 3/30/25.
//
#include "matrixParser.h"

int loadMatricesFromFileArray(const std::string &filePath, float* A, size_t A_elements, float* B, size_t B_elements) {
    std::ifstream file(filePath);
    if (!file.is_open()) {
        std::cerr << "Failed to open file " << filePath << std::endl;
        exit(EXIT_FAILURE);
    }
    std::string line;
    std::getline(file, line);
    std::istringstream issA(line);
    size_t countA = 0;
    float value;
    while (issA >> value && countA < A_elements) {
        A[countA++] = value;
    }
    std::getline(file, line);
    std::istringstream issB(line);
    size_t countB = 0;
    while (issB >> value && countB < B_elements) {
        B[countB++] = value;
    }
    return 0;
}


void parseDimensions(const std::string& filePath, size_t &m, size_t &n, size_t &k) {
    size_t pos = filePath.find_last_of("/\\");
    std::string base = pos == std::string::npos ? filePath : filePath.substr(pos + 1);
    const std::string prefix = "matrix_";
    const std::string suffix = ".txt";

    if (base.compare(0, prefix.size(), prefix) != 0 ||
        base.size() < prefix.size() + suffix.size() ||
        base.substr(base.size() - suffix.size()) != suffix)
    {
        throw std::invalid_argument("Filename does not match expected format: " + base);
    }

    std::string numbers = base.substr(prefix.size(), base.size() - prefix.size() - suffix.size());
    size_t first = numbers.find('_');
    size_t second = numbers.find('_', first + 1);
    if (first == std::string::npos || second == std::string::npos) {
        throw std::invalid_argument("Filename does not match expected format: " + base);
    }

    m = std::strtoul(numbers.substr(0, first).c_str(), nullptr, 10);
    n = std::strtoul(numbers.substr(first + 1, second - first - 1).c_str(), nullptr, 10);
    k = std::strtoul(numbers.substr(second + 1).c_str(), nullptr, 10);
}