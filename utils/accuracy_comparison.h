#ifndef ACCURACY_COMPARISON_H
#define ACCURACY_COMPARISON_H

#include <vector>
#include <string>

std::vector<std::vector<float>> loadMatrix(const std::string& filename);

std::vector<std::vector<float>> multiplyNaive(const std::vector<std::vector<float>>& A,
                                              const std::vector<std::vector<float>>& B);

float compareMatrices(const std::vector<std::vector<float>>& A,
                      const std::vector<std::vector<float>>& B);

#endif // ACCURACY_COMPARISON_H
