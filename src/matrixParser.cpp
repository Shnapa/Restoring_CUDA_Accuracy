//
// Created by gllek-pc on 3/30/25.
//
#include "matrixParser.h"

int loadMatrixFromFile(const std::string& filename, std::vector<std::vector<float>>& matrix) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file: " << filename << std::endl;
        return -1;
    }

    matrix.clear();
    std::string line;
    while (std::getline(file, line)) {
        std::istringstream ss(line);
        std::vector<float> row;
        float value;
        while (ss >> value) {
            row.push_back(value);
        }
        matrix.push_back(row);
    }

    file.close();
    return 0;
}

int loadMatricesFromFileArray(const std::string &filePath, float* A, size_t A_elements, float* B, size_t B_elements) {
    std::ifstream file(filePath);
    if (!file.is_open()) {
        std::cerr << "Error: could not open file " << filePath << std::endl;
        return -1;
    }

    std::string line;
    if (!std::getline(file, line)) {
        std::cerr << "Error: could not read the first line (Matrix A) from " << filePath << std::endl;
        return -1;
    }
    std::istringstream issA(line);
    size_t countA = 0;
    float value;
    while (issA >> value) {
        if (countA < A_elements) {
            A[countA++] = value;
        } else {
            break;
        }
    }
    if (countA != A_elements) {
        std::cerr << "Error: Expected " << A_elements << " elements for Matrix A, but found " << countA << std::endl;
        return -1;
    }

    if (!std::getline(file, line)) {
        std::cerr << "Error: could not read the second line (Matrix B) from " << filePath << std::endl;
        return -1;
    }
    std::istringstream issB(line);
    size_t countB = 0;
    while (issB >> value) {
        if (countB < B_elements) {
            B[countB++] = value;
        } else {
            break;
        }
    }
    if (countB != B_elements) {
        std::cerr << "Error: Expected " << B_elements << " elements for Matrix B, but found " << countB << std::endl;
        return -1;
    }

    return 0;
}


void parseDimensions(const std::string& filePath, size_t &m, size_t &n, size_t &k) {
    size_t pos = filePath.find_last_of("/\\");
    std::string base = pos == std::string::npos ? filePath : filePath.substr(pos + 1);
    const std::string prefix = "matrix_";
    const std::string suffix = ".txt";

    if (base.compare(0, prefix.size(), prefix) != 0 ||
        base.size() < prefix.size() + suffix.size() ||
        base.substr(base.size() - suffix.size()) != suffix)
    {
        throw std::invalid_argument("Filename does not match expected format: " + base);
    }

    std::string numbers = base.substr(prefix.size(), base.size() - prefix.size() - suffix.size());
    size_t first = numbers.find('_');
    size_t second = numbers.find('_', first + 1);
    if (first == std::string::npos || second == std::string::npos) {
        throw std::invalid_argument("Filename does not match expected format: " + base);
    }

    m = std::strtoul(numbers.substr(0, first).c_str(), nullptr, 10);
    n = std::strtoul(numbers.substr(first + 1, second - first - 1).c_str(), nullptr, 10);
    k = std::strtoul(numbers.substr(second + 1).c_str(), nullptr, 10);
}

// bool verifySampledElements(const std::vector<float>& h_A,
//                            const std::vector<float>& h_B,
//                            const std::vector<float>& h_C,
//                            int m, int n, int k,
//                            int numSamples = 10,
//                            float epsilon = 1e-5f)
// {
//     std::srand(static_cast<unsigned>(std::time(nullptr)));
//     for (int sample = 0; sample < numSamples; ++sample) {
//         int i = std::rand() % m;
//         int j = std::rand() % k;
//         float sum = 0.0f;
//         for (int l = 0; l < n; ++l) {
//             sum += h_A[i * n + l] * h_B[l * k + j];
//         }
//         if (std::fabs(sum - h_C[i * k + j]) > epsilon) {
//             std::cout << "Mismatch at (" << i << ", " << j << "): "
//                       << "CPU = " << sum << ", GPU = " << h_C[i * k + j]
//                       << std::endl;
//             return false;
//         }
//     }
//     return true;
// }
