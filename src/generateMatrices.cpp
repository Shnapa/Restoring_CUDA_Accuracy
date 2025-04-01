//
// Created by gllek-pc on 3/30/25.
//
#include <iostream>
#include <fstream>
#include <random>
#include <thread>
#include <tuple>
#include <vector>
#include <sstream>
#include "matrixParser.h"

void generate_and_write_matrices(size_t m, size_t n, size_t p) {
    std::ostringstream filename;
    filename << "../data/" << "matrix_" << m << "_" << n << "_" << p << ".txt";
    std::ofstream outfile(filename.str());
    if (!outfile.is_open()) {
        std::cerr << "Error opening file: " << filename.str() << std::endl;
        return;
    }

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dist(1.0, 10000.0);

    size_t totalA = m * n;
    for (size_t k = 0; k < totalA; ++k) {
        outfile << dist(gen);
        if (k < totalA - 1) {
            outfile << " ";
        }
    }
    outfile << "\n";

    size_t totalB = n * p;
    for (size_t k = 0; k < totalB; ++k) {
        outfile << dist(gen);
        if (k < totalB - 1) {
            outfile << " ";
        }
    }
    outfile << "\n";

    outfile.close();
    std::cout << "Finished generating matrices for sizes ("
              << m << ", " << n << ", " << p << ") in file: "
              << filename.str() << "\n";
}

int main() {
    std::vector<std::thread> threads;
    for (const auto &sizes : matrix_sizes) {
        size_t m, n, p;
        std::tie(m, n, p) = sizes;
        threads.emplace_back(generate_and_write_matrices, m, n, p);
    }

    for (auto &t : threads) {
        if (t.joinable()) {
            t.join();
        }
    }

    std::cout << "All matrices generated.\n";
    return 0;
}
