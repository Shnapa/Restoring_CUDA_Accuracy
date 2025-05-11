//
// Created by gllek-pc on 5/11/25.
//

#ifndef COMPAREM_H
#define COMPAREM_H

#include <string>
#include <vector>

inline bool compareFloats(const float a, const float b, const float epsilon) {
    const float res = std::abs((b - a)/a);
    return res < epsilon;
}

void compare(const std::vector<float>& h_C,
             size_t m, size_t k, size_t n,
             const std::string& filePath);

void compare_half(const std::vector<float>& h_C,
                  size_t m, size_t k, size_t n,
                  const std::string& filePath);

#endif //COMPAREM_H
