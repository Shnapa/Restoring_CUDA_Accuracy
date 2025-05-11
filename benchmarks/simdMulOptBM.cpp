#include <vector>
#include "matrixParser.h"
#include <benchmark/benchmark.h>
#include "simdMM.h"

static void BM_simdMulOpt(benchmark::State &state, const std::string &filePath) {
    size_t m, k, n;
    parseDimensions(filePath, m, k, n);

    const size_t size_A = m * k;
    const size_t size_B = k * n;
    const size_t size_C = m * n;

    std::vector<float> h_A(size_A), h_B(size_B), h_C(size_C);
    loadMatrices_RR(filePath, h_A, h_B);

    for (auto _ : state) {
        simdMulOpt(h_A.data(), h_B.data(), h_C.data(), m, k, n);
        benchmark::ClobberMemory();
    }
}

int main(int argc, char** argv) {
    for (const auto & filepath : filePaths) {
        benchmark::RegisterBenchmark(filepath, [filepath](benchmark::State &state) {
            BM_simdMulOpt(state, filepath);
        });
    }
    benchmark::Initialize(&argc, argv);
    benchmark::RunSpecifiedBenchmarks();
    return 0;
}
