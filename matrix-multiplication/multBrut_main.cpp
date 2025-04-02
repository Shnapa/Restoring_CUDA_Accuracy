//
// Created by Admin on 02.04.2025.
//
#include "multBrut.h"
#include <vector>
#include <iostream>

using Matrix = std::vector<int>;

int main() {
    size_t m1 = 2, n1 = 3;
    size_t m2 = 3, n2 = 2;

    Matrix A = {
        1, 2, 3,
        4, 5, 6
    };

    Matrix B = {
        7, 8,
        9, 10,
        11, 12
    };

    Matrix result = multiplication(A, m1, n1, B, m2, n2);

    printMatrix(result, m1, n2);
    return 0;
}
