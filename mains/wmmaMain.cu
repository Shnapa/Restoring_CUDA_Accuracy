#include <fstream>
#include <string>
#include <vector>
#include <cmath>
#include "matrixParser.h"
#include "mmul.cuh"
#include "compareMM.h"

int main(const int argc, char* argv[])
{
    if (argc != 2) {
        fprintf(stderr, "Usage: %s matrix_MxKxN.txt\n", argv[0]);
        return EXIT_FAILURE;
    }
    const std::string filename = argv[1];

    size_t m, k, n;
    parseDimensions(filename, m, k, n);

    std::vector<float> A_f(m*k), B_f(n*k), C_f(m*n, 0.0f);
    loadMatrices_RR(filename, A_f, B_f);

    float exec_time = 0.0f;
    wmmaMatrixMultiply(A_f.data(), B_f.data(), C_f.data(), m, k, n,exec_time);

    std::cout << "Execution time: " << exec_time << "ms" << std::endl;

    compare(C_f, m, k, n, filename);
}
