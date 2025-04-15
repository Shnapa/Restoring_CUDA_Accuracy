//
// Created by Admin on 03.04.2025.
//
#include "testing.h"
#include "src/multBrut.cpp"
#include "src/cudaMul.cu"
#include "src/cublasMul.cu"
#include "src/cudaMulOpt.cu"
#include "src/wmma.cu"

void validate_dimensions(size_t m1, size_t n1, size_t m2, size_t n2) {
    if (n1 != m2) {
        throw std::invalid_argument("Matrix dimensions do not match for multiplication: "
                                    "the number of columns in the first matrix must equal "
                                    "the number of rows in the second matrix.");
    }
}

int main() {
    size_t m = 3, n = 2;
    size_t p = 4;

    float* matrix1 = new float[m * n];
    float* matrix2 = new float[n * p];

    std::string filepath = "matrices.txt";
    write_matrices_to_file(matrix1, matrix2, m, n, p, filepath);

    generate_matrices(m, n, matrix1);
    generate_matrices(n, p, matrix2);

    try {
        validate_dimensions(m, n, n, p);
    } catch (const std::invalid_argument& e) {
        std::cerr << e.what() << std::endl;
        delete[] matrix1;
        delete[] matrix2;
        return 1;
    }

    float* standard = new float[m * p];
    multiplication(m, n, matrix1, matrix2, standard);


    float* tested = new float[m * p];
    cudaMul()

    if (compare_matrices(standard, tested, m, p)) {
        std::cout << "\nThe matrices are equal.\n";
    } else {
        std::cout << "\nThe matrices are not equal.\n";
    }

    delete[] matrix1;
    delete[] matrix2;
    delete[] standard;
    delete[] tested;

    return 0;
}
