#include <cstdio>

#include "compareMM.h"
#include "matrixParser.h"
#include <string>
#include "mmul.cuh"
#include "restore.cu"

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

    float executionTime = 0.0f;
    wmmaRestore(A_f.data(), B_f.data(), C_f.data(), m, k, n, executionTime);
    std::cout << "Execution time: " << executionTime << "ms" << std::endl;
    compare(C_f, m, k, n, filename);
}