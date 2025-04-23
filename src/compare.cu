// compare.cu
#include <iostream>
#include <vector>
#include <cmath>
#include "matrixParser.h"

inline bool compareFloats(const float a, const float b, const float epsilon = 1e-5) {
    const float res = std::abs((b - a)/a);
    return res < epsilon;
}

void compare(const float* h_C, size_t m, size_t n, size_t k, const std::string& filePath) {
     size_t A_elements = m * n;
     size_t B_elements = n * k;
 
     auto* A = static_cast<float*>(malloc(A_elements * sizeof(float)));
     auto* B = static_cast<float*>(malloc(B_elements * sizeof(float)));
     loadMatricesFromFileArray(filePath, A, A_elements, B, B_elements);
 
     auto* C_cpu = static_cast<float*>(malloc(m * k * sizeof(float)));
 
     for (size_t i = 0; i < m; ++i) {
         for (size_t j = 0; j < k; ++j) {
             float sum = 0.0f;
             for (size_t l = 0; l < n; ++l) {
                 sum += A[i * n + l] * B[l * k + j];
             }
             C_cpu[i * k + j] = sum;
         }
     }
 
     bool match = true;
     const float epsilon = 1e-5;
     for (size_t i = 0; i < m * k; ++i) {
         if (fabs(C_cpu[i] - h_C[i]) > epsilon) {
             std::cerr << "Mismatch at index " << i << ": CPU = " << C_cpu[i]
                       << ", GPU = " << h_C[i] << std::endl;
             match = false;
             break;
         }
     }
 
     if (match) {
         std::cout << "Verification passed: CPU and GPU results match." << std::endl;
     } else {
         std::cout << "Verification failed: CPU and GPU results do not match." << std::endl;
     }
 
     free(A);
     free(B);
     free(C_cpu);
 }
