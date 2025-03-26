//
// Created by gllek-pc on 3/24/25.
//
#include <iostream>
#include <fstream>
#include <vector>
#include <random>
#include <string>
#include <sstream>
#include <filesystem>
#include <tuple>

namespace fs = std::filesystem;

std::vector<double> generateMatrix(size_t totalElements, double lowerBound, double upperBound, std::mt19937 &rng) {
    std::vector<double> matrix(totalElements);
    std::uniform_real_distribution<double> dist(lowerBound, upperBound);
    for (size_t i = 0; i < totalElements; ++i) {
        matrix[i] = dist(rng);
    }
    return matrix;
}

void writeMatrixToFile(const std::string &filename, const std::vector<double> &matrix) {
    std::ofstream ofs(filename);
    if (!ofs) {
        throw std::runtime_error("Could not open file: " + filename);
    }
    std::ostringstream oss;
    for (const auto &val : matrix) {
        oss << val << " ";
    }
    ofs << oss.str();
    ofs.close();
}

int main() {
    std::random_device rd;
    std::mt19937 rng(rd());

    const double lowerBound = 1e6;
    const double upperBound = 1e9;

    std::vector<std::tuple<size_t, size_t, size_t>> configs = {
        {100,   50,   100},
        {200,   100,  150},
        {500,   250,  300},
        {1000,  500,  800},
        {2000,  1000, 1500},
        {5000,  2500, 3000},
        {10000, 5000, 8000}
    };

    std::string outputDir = "../data/matrix_dataset";
    if (!fs::exists(outputDir)) {
        fs::create_directory(outputDir);
    }

    for (const auto &config : configs) {
        size_t n, k, m;
        std::tie(n, k, m) = config;

        std::vector<double> matrixA = generateMatrix(n * k, lowerBound, upperBound, rng);
        std::vector<double> matrixB = generateMatrix(k * m, lowerBound, upperBound, rng);

        std::string filenameA = outputDir + "/A_" + std::to_string(n) + "x" + std::to_string(k) + ".txt";
        std::string filenameB = outputDir + "/B_" + std::to_string(k) + "x" + std::to_string(m) + ".txt";
        std::cout << filenameA << std::endl;
        try {
            writeMatrixToFile(filenameA, matrixA);
            writeMatrixToFile(filenameB, matrixB);
        } catch (const std::exception &e) {
            std::cerr << "Error writing files: " << e.what() << "\n";
        }
    }

    std::cout << "Matrix generation complete.\n";
    return 0;
}
