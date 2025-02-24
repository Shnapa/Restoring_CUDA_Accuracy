//
// Created by hko on 24.02.2025.
//

#include "matrix_multiplication.h"
int main() {
    Matrix A = {{1, 2, 3}, {4, 5, 6}};
    Matrix B = {{7, 8}, {9, 10}, {11, 12}};

    try {
        Matrix result = multiplication(A, B);
        std::cout << "Result:" << std::endl;
        printing(result);
    } catch (const std::invalid_argument& e) {
        std::cerr << "Error: " << e.what() << std::endl;
    }
    return 0;
}

