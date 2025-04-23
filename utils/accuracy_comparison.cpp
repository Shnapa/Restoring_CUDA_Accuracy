//
// Created by yuliana on 22.04.25.
//

#include <vector>
#include <cmath>
#include <iostream>

double norm(const std::vector<float>& mat) {
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

    double norm_diff = norm(diff);
    double norm_ref = norm(C_ref);

    if (norm_ref == 0.0) return 0.0;

    return norm_diff / norm_ref;
}
