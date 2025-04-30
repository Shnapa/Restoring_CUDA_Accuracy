//
// Created by gllek-pc on 4/30/25.
//

#ifndef COMPARE_H
#define COMPARE_H
#include <cmath>
#include <string>
#include <vector>

inline bool compareFloats(const float a, const float b, const float epsilon) {
    const float res = std::abs((b - a)/a);
    return res < epsilon;
}

void compare(const std::vector<float>& h_C,
             size_t m, size_t k, size_t n,
             const std::string& filePath);
#endif //COMPARE_H
