#ifndef MATRIX_PARSER_HPP
#define MATRIX_PARSER_HPP

#include <vector>
#include <random>
#include <fstream>

inline std::vector<std::vector<float>> loadMatrixFromFile(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open file: " + filename);
    }

    std::vector<std::vector<float>> matrix;
    std::string line;
    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::vector<float> row;
        float value;
        while (ss >> value) {
            row.push_back(value);
        }
        matrix.push_back(row);
    }
    file.close();
    return matrix;
}

inline float* loadMatrixFromFileToArray(const std::string& filename, int& rows, int& cols) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Could not open file: " << filename << std::endl;
        exit(1);
    }

    std::vector<float> matrix_data;
    std::string line;

    if (std::getline(file, line)) {
        std::stringstream ss(line);
        float value;
        while (ss >> value) {
            matrix_data.push_back(value);
        }
        cols = matrix_data.size();
    }

    while (std::getline(file, line)) {
        std::stringstream ss(line);
        float value;
        while (ss >> value) {
            matrix_data.push_back(value);
        }
    }

    file.close();

    rows = matrix_data.size() / cols;
    if (matrix_data.size() % cols != 0) {
        std::cerr << "Matrix dimensions are inconsistent with the number of elements." << std::endl;
        exit(1);
    }

    float* matrix_array = new float[matrix_data.size()];
    std::copy(matrix_data.begin(), matrix_data.end(), matrix_array);

    return matrix_array;
}

#endif