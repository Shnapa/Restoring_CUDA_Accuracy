#include "matrixParser.h"
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <vector>
#include <filesystem>

namespace fs = std::filesystem;

std::tuple<std::string, size_t, size_t> parseFilename(const std::string &filename) {
    size_t dotPos = filename.find_last_of('.');
    std::string name = (dotPos != std::string::npos) ? filename.substr(0, dotPos) : filename;

    std::istringstream iss(name);
    std::string part;
    std::vector<std::string> parts;
    while (std::getline(iss, part, '_')) {
        parts.push_back(part);
    }

    if (parts.size() < 2) {
        throw std::runtime_error("Filename format incorrect: " + filename);
    }

    std::string matrixType = parts[0];

    size_t xPos = parts[1].find('x');
    if (xPos == std::string::npos) {
        throw std::runtime_error("Dimension format incorrect in filename: " + filename);
    }
    size_t dim1 = std::stoul(parts[1].substr(0, xPos));
    size_t dim2 = std::stoul(parts[1].substr(xPos + 1));

    return {matrixType, dim1, dim2};
}

std::vector<double> loadMatrixFromFile(const std::string &filepath, size_t expectedElements) {
    std::ifstream infile(filepath);
    if (!infile) {
        throw std::runtime_error("Cannot open file: " + filepath);
    }

    std::string line;
    std::getline(infile, line);
    std::istringstream iss(line);
    std::vector<double> data;
    double value;

    while (iss >> value) {
        data.push_back(value);
    }

    if (data.size() != expectedElements) {
        throw std::runtime_error("Unexpected number of elements in " + filepath +
                                 ". Expected " + std::to_string(expectedElements) +
                                 ", got " + std::to_string(data.size()));
    }

    return data;
}

MatrixData parseMatrix(const std::string &filepath) {
    std::string filename = fs::path(filepath).filename().string();

    auto [matrixType, dim1, dim2] = parseFilename(filename);
    size_t expectedElements = dim1 * dim2;

    std::vector<double> data = loadMatrixFromFile(filepath, expectedElements);

    MatrixData md;
    md.type = matrixType;
    md.dim1 = dim1;
    md.dim2 = dim2;
    md.data = std::move(data);

    return md;
}
