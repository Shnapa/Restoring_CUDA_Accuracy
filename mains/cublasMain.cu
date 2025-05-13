#include <cuda_fp16.h>
#include <iostream>
#include <vector>
#include "matrixParser.h"
#include "mmul.cuh"
#include "compareMM.h"
#include "timeMeasurement.h"
#include <cuda_runtime.h>

void loadMatrices_RR_half(const std::string &filePath,
                     std::vector<__half> &A,
                     std::vector<__half> &B)
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
    while (isa >> v && i < A.size()) A[i++] = __float2half(v);
    std::getline(fin, line);
    isa.clear();
    isa.str(line);
    i = 0;
    while (isa >> v && i < B.size()) B[i++] = __float2half(v);
}

int main(const int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <matrix_file_path>" << std::endl;
        return 1;
    }
    // const std::string filePath = argv[1];

    for (const auto &filepath : filePaths) {
        std::cout << filepath << std::endl;
        size_t m, k, n;
        parseDimensions(filepath, m, k, n);

        const size_t size_A = m * k;
        const size_t size_B = k * n;
        const size_t size_C = m * n;

        std::vector<__half> h_A(size_A), h_B(size_B);
        std::vector<float> h_C(size_C);
        loadMatrices_RR_half(filepath, h_A, h_B);
        float exec_time = 0.0f;

        cublasMatrixMultiplyHalf(h_A.data(), h_B.data(), h_C.data(), m, k, n, exec_time);

        // std::cout << "Execution time: " << exec_time << "ms" << std::endl;

        compare(h_C, m, k, n, filepath);

    }
    return 0;
}
