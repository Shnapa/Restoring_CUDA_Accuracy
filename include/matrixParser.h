#ifndef MATRIX_PARSER_HPP
#define MATRIX_PARSER_HPP

#include <iostream>
#include <vector>
#include <random>
#include <fstream>
#include <sstream>


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
        exit(EXIT_FAILURE);
    }

    std::vector<float> data;
    float value;
    while (file >> value) {
        data.push_back(value);
    }
    file.close();

    rows = 1;
    cols = data.size();

    float* array = new float[data.size()];
    std::copy(data.begin(), data.end(), array);
    return array;
}
#endif
