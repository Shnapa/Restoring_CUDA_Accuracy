#include <iostream>
#include <vector>
#include "matrixParser.h"
#include "mmul.h"
#include "compare.cu"
int main(const int argc, char** argv) {
    if (argc < 2) {
        std::cerr << argv[0] << " <matrix_file_path>\n";
        return 1;
    }
    const std::string filePath = argv[1];

    size_t m, k, n;
    parseDimensions(filePath, m, k, n);

    const size_t size_A = m * k;
    const size_t size_B = k * n;
    const size_t size_C = m * n;

    std::vector<float> h_A(size_A), h_B(size_B), h_C(size_C);
    loadMatrices_RR(filePath, h_A, h_B);

    float exec_time = 0.0f;
    cudaMatrixMultiply(h_A.data(), h_B.data(), h_C.data(), m, k, n, exec_time);
    std::cout << "Execution time: " << exec_time << "ms" << std::endl;

    compare(h_C, m, k, n, filePath);
}
