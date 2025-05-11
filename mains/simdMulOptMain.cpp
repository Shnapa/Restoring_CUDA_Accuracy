#include <vector>

#include "matrixParser.h"
#include "timeMeasurement.h"
#include "simdMM.h"
#include "compareMM.h"

int main(const int argc, char** argv) {
    if(argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <matrix_file_path>" << std::endl;
        return 1;
    }
    const std::string filePath = argv[1];

    size_t m, n, k;
    parseDimensions(filePath, m, k, n);

    const size_t size_A = m * k;
    const size_t size_B = k * n;
    const size_t size_C = m * n;

    std::vector<float> h_A(size_A), h_B(size_B), h_C(size_C);
    loadMatrices_RR(filePath, h_A, h_B);

    const auto start = get_current_time_fenced();
    simdMulOpt(h_A.data(), h_B.data(), h_C.data(), m, k, n);
    const auto end = get_current_time_fenced();
    std::cout << "Execution time: " << to_ms(end - start) << "ms" << std::endl;

    compare(h_C, m, k, n, filePath);
}
