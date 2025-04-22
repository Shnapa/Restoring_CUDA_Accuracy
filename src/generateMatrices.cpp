#include <iostream>
#include <fstream>
#include <random>
#include <thread>
#include <tuple>
#include <vector>
#include <string>
#include "matrixParser.h"

void generate_and_write_matrices(size_t m, size_t n, size_t k) {
    std::string filename = "../data/matrix_" + std::to_string(m) + "_" + std::to_string(n) + "_" + std::to_string(k) + ".txt";
    std::ofstream outfile(filename);
    if (!outfile) {
        std::cerr << "Error opening file: " << filename << "\n";
        return;
    }

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dist(-10000.0, 10000.0);

    size_t totalA = m * n;
    for (size_t i = 0; i < totalA; ++i) {
        outfile << dist(gen);
        if (i + 1 < totalA)
            outfile << ' ';
    }
    outfile << '\n';

    size_t totalB = n * k;
    for (size_t i = 0; i < totalB; ++i) {
        outfile << dist(gen);
        if (i + 1 < totalB)
            outfile << ' ';
    }
    outfile << '\n';

    outfile.close();
    std::cout << "Finished generating matrices for (" << m << ", " << n << ", " << k << ") in file: " << filename << "\n";
}

int main() {
    std::vector<std::thread> threads;
    for (const auto &sizes : matrix_sizes) {
        size_t m, n, k;
        std::tie(m, n, k) = sizes;
        threads.emplace_back(generate_and_write_matrices, m, n, k);
    }
    for (auto &t : threads) {
        if (t.joinable()) {
            t.join();
        }
    }
    std::cout << "All matrices generated.\n";
    return 0;
}
