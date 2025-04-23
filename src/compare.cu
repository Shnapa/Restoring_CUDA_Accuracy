// compare.cu
 #include <iostream>
 #include <vector>
 #include <cmath>
 #include "matrixParser.h"
 
 inline bool compareFloats(const float a, const float b, const float epsilon = 1e-5) {
     const float res = std::abs((b - a)/a);
     return res < epsilon;
 }
 
 void compare(const std::vector<float>& h_C,
              const size_t m, const size_t k, const size_t n,
              const std::string& filePath)
 {
     const size_t size_A = m * k;
     const size_t size_B = k * n;
     const size_t size_C = m * n;
 
     std::vector<float> A(size_A), B(size_B), C_cpu(size_C);
     loadMatrices_RR(filePath, A, B);
 
     for (size_t i = 0; i < m; ++i) {
         for (size_t j = 0; j < n; ++j) {
             float sum = 0.0f;
             for (size_t l = 0; l < k; ++l) {
                 sum += A[i * k + l] * B[l * n + j];
             }
             C_cpu[i * n + j] = sum;
         }
     }
 
     constexpr float epsilon = 1e-5f;
     bool match = true;
     for (size_t idx = 0; idx < size_C; ++idx) {
         if (!compareFloats(C_cpu[idx], h_C[idx])) {
             std::cerr << "Mismatch at idx " << idx
                       << ": CPU=" << C_cpu[idx]
                       << " GPU=" << h_C[idx] << "\n";
             match = false;
             break;
         }
     }
 
     if (match) {
         std::cout << "Verification passed: CPU and GPU results match.\n";
     } else {
         std::cout << "Verification failed.\n";
     }
 }
