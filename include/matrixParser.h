#ifndef MATRIX_PARSER_HPP
#define MATRIX_PARSER_HPP

#include <string>
#include <vector>
#include <tuple>

struct MatrixData {
    std::string type;    // Matrix type: "A" or "B"
    size_t dim1;         // First dimension (rows for matrix A, etc.)
    size_t dim2;         // Second dimension (columns)
    std::vector<double> data;  // Matrix values stored in row-major order
};

std::tuple<std::string, size_t, size_t> parseFilename(const std::string &filename);

std::vector<double> loadMatrixFromFile(const std::string &filepath, size_t expectedElements);

MatrixData parseMatrix(const std::string &filepath);

#endif // MATRIX_PARSER_HPP
