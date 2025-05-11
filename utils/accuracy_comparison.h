#pragma once
#include <vector>
#include <string>

std::vector<std::vector<float>> loadMatrix(const std::string& filename);
std::vector<std::vector<float>> multiplyNaive(const std::vector<std::vector<float>>& A,
                                              const std::vector<std::vector<float>>& B);

std::vector<double> referenceGEMM_FP64(const std::vector<float>& A, const std::vector<float>& B,
                                       size_t m, size_t k, size_t n);

double relativeResidual(const std::vector<double>& C_ref, const std::vector<float>& C_target);
