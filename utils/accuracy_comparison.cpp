//
// Created by yuliana on 22.04.25.
//

#include <vector>
#include <cmath>
#include <iostream>

double frobeniusNorm(const std::vector<float>& mat) {
    double sum = 0.0;
    for (float val : mat) {
        sum += static_cast<double>(val) * val;
    }
    return std::sqrt(sum);
}

double relativeResidual(const std::vector<float>& C_ref, const std::vector<float>& C_target) {
    if (C_ref.size() != C_target.size()) {
        throw std::runtime_error("Matrix sizes do not match");
    }

    std::vector<float> diff(C_ref.size());
    for (size_t i = 0; i < C_ref.size(); ++i) {
        diff[i] = C_ref[i] - C_target[i];
    }

    double norm_diff = frobeniusNorm(diff);
    double norm_ref = frobeniusNorm(C_ref);

    if (norm_ref == 0.0) return 0.0;

    return norm_diff / norm_ref;
}

int main(int argc, char** argv) {
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " A.txt B.txt\n";
        return 1;
    }

    return 0;
}

